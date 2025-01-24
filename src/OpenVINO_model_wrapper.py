"""
Courtesy to: https://github.com/openvinotoolkit/openvino_notebooks/blob/latest/notebooks/person-tracking-webcam/person-tracking.ipynb
"""

import openvino as ov

core = ov.Core()


class OpenVINO_model_wrapper:

    def __init__(self, model_path, batchsize=1, device="AUTO"):
        """
        Initialize the model object

        Parameters
        ----------
        model_path: path of inference model
        batchsize: batch size of input data
        device: device used to run inference
        """
        self.model = core.read_model(model=model_path)
        self.input_layer = self.model.input(0)
        self.input_shape = self.input_layer.shape
        self.height = self.input_shape[2]
        self.width = self.input_shape[3]

        for layer in self.model.inputs:
            input_shape = layer.partial_shape
            input_shape[0] = batchsize
            self.model.reshape({layer: input_shape})
        self.compiled_model = core.compile_model(model=self.model, device_name=device)
        self.output_layer = self.compiled_model.output(0)

    def predict(self, input):
        """
        Run inference

        Parameters
        ----------
        input: array of input data
        """
        result = self.compiled_model(input)[self.output_layer]
        return result