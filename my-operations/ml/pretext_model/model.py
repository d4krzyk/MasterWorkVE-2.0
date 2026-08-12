"""Siec zadania pretekstowego -- specyfikacja z suplementu Gao (§I), doslownie.

    galaz audio    Echo-Net = `SimpleAudioDepthNet` Paridy BEZ dekodera:
                   trzy konwolucje 8x8 / 4x4 / 3x3, BatchNorm + ReLU po kazdej,
                   warstwa liniowa (conv1x1) redukujaca do 512.
    galaz wizualna enkoder sieci DOCELOWEJ. Dla RGB2Depth na Replice Gao uzywa
                   enkodera U-Netu (5 warstw, 128x128x3 -> 4x4x512) -- czyli
                   dokladnie `rgbdepth_convlayer1..5` z `RGBDepthNet` Paridy.
                   NIE ResNet-50: ten jest u Gao wylacznie dla NYU-V2/DIODE.
                   Na koncu conv1x1 redukujacy kanaly, potem splaszczenie.
    fuzja          konkatenacja cech audio i wizualnych -> warstwa w pelni
                   polaczona + ReLU, redukcja do D = 128.
    glowa          jedna warstwa w pelni polaczona -> K klas.
    strata         cross-entropy (plaska).

ZERO ZMIAN W KODZIE PARIDY. Obie galezie to jego oryginalne moduly, budowane
przez jego `ModelBuilder` -- uzywamy tylko ich CZESCI ENKODUJACEJ, wywolujac
warstwy bezposrednio. Dekodery istnieja w pamieci, nie dostaja gradientu i nie
wchodza do przeniesienia.

DLACZEGO PELNY `RGBDepthNet`, A NIE SAM ENKODER. Zadaniem docelowym jest ten sam
`RGBDepthNet`, wiec budujac tu ten sam modul mamy pewnosc, ze inicjalizacja,
nazwy w `state_dict` i ksztalty sa identyczne -- przeniesienie wag to wtedy
zwykle `load_state_dict`, a nie recznie sklejane mapowanie nazw, ktore moze sie
cicho rozjechac. `conv1x1` redukujacy kanaly jest u Paridy zakomentowany
(`networks.py:172`, `create_conv(512, 8, 1, 0)`) -- odtwarzamy go tutaj, w
NASZYM module, zamiast odkomentowywac w jego pliku.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .. import paths

# Domyslna liczba kanalow, do ktorej conv1x1 redukuje cechy wizualne. Wartosc
# z zakomentowanej linii Paridy: 512 -> 8, wiec cecha wizualna to 4*4*8 = 128.
VISUAL_REDUCE_CH = 8
FUSION_DIM = 128
AUDIO_FEATURE_LENGTH = 512   # `build_audiodepth` Paridy: audio_feature_length=512


class OrientationPretextNet(nn.Module):
    """(obraz z orientacji i, echo z orientacji j) -> klasa przesuniecia (j - i)."""

    def __init__(self, n_classes: int, audio_shape=(2, 257, 166),
                 visual_reduce: int = VISUAL_REDUCE_CH, fusion_dim: int = FUSION_DIM):
        super().__init__()
        paths.add_parida_to_syspath()
        from models.models import ModelBuilder
        from models.networks import create_conv, weights_init

        b = ModelBuilder()
        # Te same dwa modulu, ktore buduje Model 1 -- ta sama inicjalizacja.
        self.net_rgbdepth = b.build_rgbdepth()
        self.net_audiodepth = b.build_audiodepth(audio_shape=list(audio_shape))

        self.visual_conv1x1 = create_conv(512, visual_reduce, 1, 0)
        visual_dim = visual_reduce * 4 * 4          # enkoder konczy na 4x4
        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + AUDIO_FEATURE_LENGTH, fusion_dim), nn.ReLU())
        self.head = nn.Linear(fusion_dim, n_classes)

        for m in (self.visual_conv1x1, self.fusion, self.head):
            m.apply(weights_init)

        self.n_classes = n_classes
        self.visual_dim = visual_dim

    # ------------------------------------------------------------- enkodery

    def encode_visual(self, img: torch.Tensor) -> torch.Tensor:
        n = self.net_rgbdepth
        x = n.rgbdepth_convlayer1(img)
        x = n.rgbdepth_convlayer2(x)
        x = n.rgbdepth_convlayer3(x)
        x = n.rgbdepth_convlayer4(x)
        x = n.rgbdepth_convlayer5(x)      # (B, 512, 4, 4)
        x = self.visual_conv1x1(x)        # (B, 8, 4, 4)
        return x.flatten(1)

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        n = self.net_audiodepth
        x = n.feature_extraction(audio)
        x = x.view(x.shape[0], -1, 1, 1)
        x = n.conv1x1(x)                  # (B, 512, 1, 1)
        return x.flatten(1)

    def forward(self, batch: dict) -> torch.Tensor:
        v = self.encode_visual(batch["img"])
        a = self.encode_audio(batch["audio"])
        f = self.fusion(torch.cat([v, a], dim=1))
        return self.head(f)

    # ------------------------------------------------------------ przeniesienie

    def encoder_state_dict(self) -> dict:
        """Wagi, ktore ida do zadania docelowego: CALY `RGBDepthNet`.

        Zapisujemy caly modul, a nie same warstwy enkodera, zeby plik dalo sie
        wczytac jednym `load_state_dict`; `transfer.py` i tak bierze z niego
        wylacznie klucze `rgbdepth_convlayer*` -- dekoder zadania docelowego
        musi startowac swiezy, bo tego wlasnie znaczy "pretrenowany ENKODER".
        """
        return self.net_rgbdepth.state_dict()


def load_pretrained_encoder(net_rgbdepth: nn.Module, weights_path, *,
                            strict_prefix: str = "rgbdepth_convlayer") -> dict:
    """Wstawia pretrenowany enkoder do swiezego `RGBDepthNet`. Zwraca raport.

    Raport jest zwracany, a nie logowany, bo liczba dopasowanych kluczy jest
    dowodem, ze przeniesienie w ogole zaszlo. Ciche `strict=False`, ktore nie
    dopasowalo NICZEGO, wygladalo by dokladnie tak samo jak `Scratch` -- i cala
    tabela 4.5 bylaby wtedy tabela pieciu razy tego samego warunku.
    """
    src = torch.load(weights_path, map_location="cpu", weights_only=True)
    enc = {k: v for k, v in src.items() if k.startswith(strict_prefix)}
    if not enc:
        raise RuntimeError(f"{weights_path}: brak kluczy zaczynajacych sie od "
                           f"{strict_prefix!r} -- to nie sa wagi enkodera")
    tgt = net_rgbdepth.state_dict()
    shape_ok = {k: v for k, v in enc.items() if k in tgt and tgt[k].shape == v.shape}
    missing_shape = sorted(set(enc) - set(shape_ok))
    net_rgbdepth.load_state_dict(shape_ok, strict=False)
    return {
        "weights": str(weights_path),
        "n_encoder_keys_in_file": len(enc),
        "n_loaded": len(shape_ok),
        "n_shape_mismatch": len(missing_shape),
        "shape_mismatch_keys": missing_shape,
        "n_target_keys_total": len(tgt),
        "ok": bool(shape_ok and not missing_shape),
    }
