# Third-party software and licenses

This project uses the following external software:

## Habitat-Sim

Source:
https://github.com/facebookresearch/habitat-sim

License:
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

The original license file is preserved in:
`habitat-sim/LICENSE`


## Habitat-Lab

Source:
https://github.com/facebookresearch/habitat-lab

License:
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

The original license file is preserved in:
`habitat-lab/LICENSE`


## SoundSpaces

Source:
https://github.com/facebookresearch/sound-spaces

License:
Creative Commons Attribution 4.0 International

The original license file is preserved in:
`sound-spaces/LICENSE`


## Beyond Image to Depth (Parida et al., CVPR 2021)

Source:
https://github.com/krantiparida/beyond-image-to-depth

Pinned upstream commit:
`dcdef5122fa456a92bd58ead4eea0a777158c535` (see `beyond-image-to-depth/COMMIT_HASH.txt`)

License:
MIT License

Copyright (c) 2021 Kranti Kumar Parida

The original license file is preserved in:
`beyond-image-to-depth/LICENSE`

Used as the reference implementation for audio-visual depth prediction
(network architecture, loss, optimizer). Its files are vendored **unmodified** —
this project's training code lives separately in `my-operations/ml/` and imports
the networks rather than editing them, so that the independent variable of the
study (angular density of echo sampling) is not confounded by changes to the
reference model.


Modifications made as part of this project are Copyright (c) Daniel Krzykowski.
