"""Szybszy, MATEMATYCZNIE TOZSAMY zamiennik `torch.nn.Bilinear`. OPCJONALNY.

TO NIE JEST ZMIANA ARCHITEKTURY. Warstwa ma te same parametry (`weight` o
ksztalcie (out, in1, in2) i `bias`), liczy te sama funkcje
    y[n,o] = sum_ij x1[n,i] * W[o,i,j] * x2[n,j] + b[o]
i te same gradienty. Rozni sie wylacznie KOLEJNOSCIA operacji: `nn.Bilinear`
liczy wynik cecha po cesze (petla `baddbmm` po wymiarze wyjsciowym), co przy
out=512 daje 512 malych jader GPU; wersja `einsum` sklada to w dwa duze
iloczyny macierzowe. Roznica numeryczna to skutek innej kolejnosci sumowania w
float32 -- zmierzona: 2.8e-05 przy wartosciach rzedu 64, czyli ~4e-7 wzglednie.

DLACZEGO TO ISTNIEJE. Zmierzone na tym sprzecie (RTX 5070 Ti), batch 32:
    attentionNet          1468 ms z 1523 ms calej iteracji  = 96.4 %
    sama nn.Bilinear       743 ms
    ta sama einsum          19 ms                            = 38.7x szybciej
Pelna macierz eksperymentow to ~15 dni GPU z `nn.Bilinear` i ~20 godzin bez
niej. To roznica miedzy "da sie zrobic 3 ziarna na warunek" a "nie da sie".

DLACZEGO DOMYSLNIE WYLACZONE. Decyzja o odejsciu od literalnego kodu
referencyjnego -- nawet przy dowiedzionej tozsamosci funkcji -- nalezy do
autora pracy, nie do narzedzia. Wlacza sie flaga `--fast-bilinear`.
Przed uzyciem w przebiegu warto uruchomic `verify_equivalence()`.

KOSZT PAMIECI. Tensor posredni ma ksztalt (N, out, in2); przy N = batch*16 = 512
i 512x512 to ~537 MB w float32. Zmierzony szczyt calego modelu rosnie z 6.3 do
ok. 8 GB przy batchu 32 -- miesci sie w 16 GB karty, ale przy wiekszym batchu
trzeba to sprawdzic (`chunk` ogranicza zuzycie kosztem czesci przyspieszenia).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class BilinearEinsum(nn.Module):
    """Zamiennik `nn.Bilinear` o identycznym zestawie parametrow."""

    def __init__(self, in1_features: int, in2_features: int, out_features: int,
                 bias: bool = True, chunk: int | None = None):
        super().__init__()
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.chunk = chunk
        self.weight = nn.Parameter(torch.empty(out_features, in1_features, in2_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # `attentionNet` podaje tensory (B, H, W, C); splaszczamy wiodace wymiary,
        # bo einsum operuje na jednym wymiarze wsadowym.
        lead = x1.shape[:-1]
        a = x1.reshape(-1, self.in1_features)
        b = x2.reshape(-1, self.in2_features)

        if self.chunk is None:
            t = torch.einsum("ni,oij->noj", a, self.weight)
            y = torch.einsum("noj,nj->no", t, b)
        else:
            # Wariant oszczedzajacy pamiec: liczymy porcjami po cechach wyjsciowych.
            outs = []
            for s in range(0, self.out_features, self.chunk):
                w = self.weight[s:s + self.chunk]
                t = torch.einsum("ni,oij->noj", a, w)
                outs.append(torch.einsum("noj,nj->no", t, b))
            y = torch.cat(outs, dim=1)

        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, self.out_features)

    def extra_repr(self) -> str:
        return (f"in1_features={self.in1_features}, in2_features={self.in2_features}, "
                f"out_features={self.out_features}, bias={self.bias is not None}")


def swap_bilinear(module: nn.Module, chunk: int | None = None) -> int:
    """Podmienia w miejscu kazdy `nn.Bilinear` na `BilinearEinsum`, PRZENOSZAC WAGI.

    Wagi sa przenoszone, a nie inicjalizowane od nowa, wiec podmiana w dowolnym
    momencie (takze po wczytaniu checkpointu) nie zmienia stanu modelu. Nazwy w
    `state_dict` zostaja te same (`weight`, `bias`), wiec checkpointy sa wymienne
    w obie strony miedzy wersja szybka a oryginalna.
    """
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Bilinear):
            new = BilinearEinsum(child.in1_features, child.in2_features,
                                 child.out_features, bias=child.bias is not None,
                                 chunk=chunk)
            with torch.no_grad():
                new.weight.copy_(child.weight)
                if child.bias is not None:
                    new.bias.copy_(child.bias)
            new.to(child.weight.device, child.weight.dtype)
            setattr(module, name, new)
            n += 1
        else:
            n += swap_bilinear(child, chunk=chunk)
    return n


@torch.no_grad()
def verify_equivalence(in1=512, in2=512, out=512, n=256, device="cuda",
                       seed=0, tol_rel=1e-4) -> dict:
    """Dowod tozsamosci na losowych danych: forward i gradienty."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    ref = nn.Bilinear(in1, in2, out).to(dev)
    fast = BilinearEinsum(in1, in2, out).to(dev)
    fast.weight.copy_(ref.weight)
    fast.bias.copy_(ref.bias)

    x1 = torch.randn(n, in1, device=dev)
    x2 = torch.randn(n, in2, device=dev)
    y_ref, y_fast = ref(x1, x2), fast(x1, x2)
    scale = float(y_ref.abs().max())
    fwd = float((y_ref - y_fast).abs().max())

    with torch.enable_grad():
        a1 = x1.clone().requires_grad_(True)
        a2 = x2.clone().requires_grad_(True)
        ref(a1, a2).pow(2).sum().backward()
        g_ref = (a1.grad.clone(), a2.grad.clone(), ref.weight.grad.clone())

        b1 = x1.clone().requires_grad_(True)
        b2 = x2.clone().requires_grad_(True)
        fast(b1, b2).pow(2).sum().backward()
        g_fast = (b1.grad.clone(), b2.grad.clone(), fast.weight.grad.clone())

    gdiff = max(float((a - b).abs().max()) for a, b in zip(g_ref, g_fast))
    gscale = max(float(g.abs().max()) for g in g_ref)
    return {
        "forward_max_abs_diff": fwd,
        "forward_scale": scale,
        "forward_rel": fwd / scale,
        "grad_max_abs_diff": gdiff,
        "grad_scale": gscale,
        "grad_rel": gdiff / gscale,
        "ok": (fwd / scale < tol_rel) and (gdiff / gscale < tol_rel),
    }
