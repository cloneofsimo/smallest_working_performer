# smallest_working_performer

Performer and its application on Vision task

## Linear - time transformer + ViT = Profit?

Implementation of performer from :  https://arxiv.org/pdf/2009.14794.pdf

Example usage :

```python
from model import ViP
model = ViP(
    image_pix = 28,
    patch_pix = 2, # this will result in 14 * 14 words
    class_cnt = 10,
    layer_cnt = 3,
    kernel_ratio = 0.8
)
```
Example train.py file contains simple MNIST example.

Also, my implementation is short and easy to understand. You're welcome.
