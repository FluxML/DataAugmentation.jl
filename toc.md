

## DataAugmentation.jl

|                                                           |                                  |
| :-------------------------------------------------------- | -------------------------------: |
| [Introduction](docs/literate/intro.md)                    |                                  |
| [Item interface](docs/literate/iteminterface.md)          |         [`base.jl`](src/base.jl) |
| [Transformation interface](docs/literate/tfminterface.md) |         [`base.jl`](src/base.jl) |
| [Buffering](docs/literate/buffering.md)                   | [`buffered.jl`](src/buffered.jl) |

### Projective transformations

Augmenting spatial data like images and keypoints.

|                                                    |                                                                                                                                                          |
| :------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------: |
| [Introduction](docs/literate/projective/intro.md)  |                                                                                                                                                          |  |
| [Interface](docs/literate/projective/interface.md) |                                            [`projective/base.jl`](src/projective/base.jl)           , [`projective/bounds.jl`](src/projective/bounds.jl) |  |
| [Spatial data](docs/literate/projective/data.md)   |                             [`items/image.jl`](src/items/image.jl), [`items/mask.jl`](src/items/mask.jl), [`items/keypoints.jl`](src/items/keypoints.jl) |  |
| [Usage](docs/literate/projective/usage.md)         | [`projective/compose.jl`](src/projective/compose.jl), [`projective/crop.jl`](src/projective/crop.jl), [`projective/affine.jl`](src/projective/affine.jl) |
| [Gallery](docs/literate/projective/gallery.jl)     |                                                                                                                                                          |

### Preprocessing

Transformations for getting data in and out of a model.

|                                            |                                           |
| :----------------------------------------- | ----------------------------------------: |
| [Overview](docs/literate/preprocessing.md) | [`preprocessing.jl`](src/preprocessing.jl) |

[Reference](docstrings.md)
