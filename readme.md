Implementation of the transformer architecture from "Attention is All You Need" in PyTorch, tested on a simple addition task.

<img src="https://camo.githubusercontent.com/8e489fab63c274c0dbbd3e882c0b9044f74392a1c0bda92393839796d44d621f/687474703a2f2f696d6775722e636f6d2f316b72463252362e706e67" alt="img" style="zoom: 67%;" />

The model is given a string of form `A+B` and should predict the result of the expression. For example

```
Input:  1234+99887
Output: 101121
```

When trained on random expressions of length up to 10, for 15 minutes, the model achieves a digit accuracy of around a 70%.

## Citation

[1] "Attention is All You Need" (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017)