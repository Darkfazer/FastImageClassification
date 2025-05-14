import unittest
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
import sys
import os
import tempfile


from model_builder import get_model_classification

class TestModelClassification(unittest.TestCase):
    
    def test_mobilenet_binary_classification(self):
        """Test if the function creates a proper MobileNetV2 model for binary classification."""
        model = get_model_classification(
            input_shape=(224, 224, 3),
            model="mobilenet",
            weights=None,  # Using None to speed up test execution
            n_classes=1,
            multi_class=False
        )
        
        # Check if model is a Keras Model
        self.assertIsInstance(model, Model)
        
        # Check input shape
        self.assertEqual(model.input_shape, (None, 224, 224, 3))
        
        # Check output shape for binary classification
        self.assertEqual(model.output_shape, (None, 1))
        
        # Check if output activation is sigmoid
        self.assertEqual(model.layers[-1].activation.__name__, "sigmoid")
        
        # Check loss function
        self.assertEqual(model.loss, tf.keras.losses.binary_crossentropy)
        
    def test_resnet_binary_classification(self):
        """Test if the function creates a proper ResNet50 model for binary classification."""
        model = get_model_classification(
            input_shape=(224, 224, 3),
            model="resnet",
            weights=None,  # Using None to speed up test execution
            n_classes=1,
            multi_class=False
        )
        
        # Check if model is a Keras Model
        self.assertIsInstance(model, Model)
        
        # Check if base model is ResNet50
        # Find the layer that should be ResNet50
        resnet_layer = None
        for layer in model.layers:
            if "resnet" in layer.name.lower():
                resnet_layer = layer
                break
        
        self.assertIsNotNone(resnet_layer, "ResNet50 layer not found in the model")
    
    def test_multi_class_classification(self):
        """Test if the function creates a proper model for multi-class classification."""
        n_classes = 10
        model = get_model_classification(
            input_shape=(224, 224, 3),
            model="mobilenet",
            weights=None,  # Using None to speed up test execution
            n_classes=n_classes,
            multi_class=True
        )
        
        # Check output shape for multi-class classification
        self.assertEqual(model.output_shape, (None, n_classes))
        
        # Check if output activation is softmax
        self.assertEqual(model.layers[-1].activation.__name__, "softmax")
        
        # Check loss function
        self.assertEqual(model.loss, tf.keras.losses.categorical_crossentropy)
    
    def test_custom_input_shape(self):
        """Test if the function handles custom input shapes correctly."""
        custom_shape = (320, 320, 3)
        model = get_model_classification(
            input_shape=custom_shape,
            model="mobilenet",
            weights=None,
            n_classes=4,
            multi_class=False
        )
        
        # Check if input shape is respected
        self.assertEqual(model.input_shape, (None,) + custom_shape)
    
    def test_model_inference(self):
        """Test if the model can perform inference on sample data."""
        input_shape = (224, 224, 3)
        n_classes = 4
        model = get_model_classification(
            input_shape=input_shape,
            model="mobilenet",
            weights=None,
            n_classes=n_classes,
            multi_class=False
        )
        
        # Create a batch of random images
        batch_size = 2
        sample_images = np.random.random((batch_size,) + input_shape)
        
        # Perform inference
        predictions = model.predict(sample_images)
        
        # Check output shape
        self.assertEqual(predictions.shape, (batch_size, n_classes))
        
        # For sigmoid activation, all values should be between 0 and 1
        self.assertTrue(np.all(predictions >= 0))
        self.assertTrue(np.all(predictions <= 1))
    
    def test_model_training(self):
        """Test if the model can be trained with sample data."""
        input_shape = (224, 224, 3)
        n_classes = 4
        batch_size = 2
        
        # Create a small model for faster testing
        model = get_model_classification(
            input_shape=input_shape,
            model="mobilenet",
            weights=None,
            n_classes=n_classes,
            multi_class=False
        )
        
        # Create random training data
        X_train = np.random.random((batch_size,) + input_shape)
        y_train = np.random.random((batch_size, n_classes))
        
        # Train for just one step to verify it works
        history = model.fit(
            X_train, y_train,
            epochs=1,
            verbose=0
        )
        
        # Check if training happened without errors
        self.assertIsNotNone(history)
        self.assertTrue('loss' in history.history)
    
    def test_model_save_load(self):
        """Test if the model can be saved and loaded correctly."""
        model = get_model_classification(
            input_shape=(224, 224, 3),
            model="mobilenet",
            weights=None,
            n_classes=4,
            multi_class=False
        )
        
        # Create a temporary directory for saving the model
        with tempfile.TemporaryDirectory() as tmpdirname:
            model_path = os.path.join(tmpdirname, 'test_model.h5')
            
            # Save the model
            model.save(model_path)
            
            # Check if the file exists
            self.assertTrue(os.path.exists(model_path))
            
            # Load the model
            loaded_model = tf.keras.models.load_model(model_path)
            
            # Check if loaded model has the same structure
            self.assertEqual(model.input_shape, loaded_model.input_shape)
            self.assertEqual(model.output_shape, loaded_model.output_shape)


if __name__ == '__main__':
    unittest.main()