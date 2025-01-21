# Setup

### Create EvalModel
1. Duplicate `eval_model.py`.
2. Implement the `__init__` function. This is the function for **initializing** the model.
3. Implement the `predict` function. The input is **the file path**, and the output should be the **likelihood** of fake. Add any code used for predicting here, such as loading the input file, preprocessing, etc.

### Edit evaluate.py
1. Modify the arguments for the `EvalModel` if you added additional arguments to the `EvalModel` constructor.
2. Add additional CLI arguments.

# Run
Run from the root directory of your project. Dataset has to be **structured** like this:

<pre>
    dataset:
        - fake
        - real
</pre>

```bash
python eval_model/evaluate.py -d <path to Data Directory> <any other additional arguments>
