# FAQs

1. Can I run GaMPEN on any galaxy image?

    No! Please see our recommendations in the [Using GaMPEN](Using_GaMPEN).


2. I am having difficulty enabling GPU support. What should I do?

    Try using Google Colab like we have done in the [Tutorials](Tutorials). 


3. I am on a multi-GPU machine; and my code hangs during training or inference.
    
    GaMPEN uses `torch.nn.DataParallel` for it's multi-GPU training and inference and this might not work in certain systems and cause your program to hang (without any output) right at the point where the model interacts with the GPU in any way.

    If this is the case, the easiest solution is to set the `CUDA_VISIBLE_DEVICES` environment variable in your shell to one of the available GPUs. For example, if there are two GPUs, these would be `0` and `1`; and we recommend setting `CUDA_VISIBLE_DEVICES` to either `0` or `1` in this scenario.


4. What if my question is not answered here?

    Please send me an e-mail at this ``aritraghsh09@xxxxx.com`` GMail address. Additionally, if you have spotted a bug in the code/documentation or you want to propose a new feature, please feel free to open an issue/a pull request on [GitHub](https://github.com/aritraghsh09/GaMorNet).

