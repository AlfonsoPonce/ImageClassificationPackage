'''
Class that represents model storage.

Author: Alfonso Ponce Navarro
Date: 05/11/2023
'''

from .model_repo import tensorflow_example


class Zoo():
    def __init__(self, num_classes: int):
        '''
        Instantiate Zoo object.
        :param num_classes: number of classes to be detected
        '''
        # add background class
        self.num_classes = num_classes + 1

    def get_model(self, name: str, new_head:bool):
        '''
        Function to select detection model
        :param name: name of the model to be used. The name must be the same as the filename in model_repo/ folder.
        :param new_head: If True, then the pretrained head is replaced by a not trained one.
        :return: torch object detection.
        '''
        model = None
        if name == 'tensorflow_example':
            model = tensorflow_example.create_model(self.num_classes)


        return model
