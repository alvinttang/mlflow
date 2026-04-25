from custom_model.relative_import_test.sub_class import SubClass

from mlflow.pyfunc import PythonModel


class ModelWithRelativeImport(PythonModel):
    def __init__(self):
        self.sub = SubClass()

    def predict(self, context, model_input, params=None):
        return [self.sub.value()] * len(model_input)
