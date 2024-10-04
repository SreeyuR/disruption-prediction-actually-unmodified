import unittest
import torch
from torch.nn import functional as F
from transformers import TrainingArguments, Trainer
from evaluation import MultiLossExperimentalTrainer
import evaluation
import numpy as np

class TestMultiLossExperimentalTrainer(unittest.TestCase):

    def test_compute_loss(self):
        # Create a mock model

        class MockOutput:
            def __init__(self, logits):
                self.logits = logits

        class MockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()

            def forward(self, input_ids):
                batch_size, seq_len, num_params = input_ids.size()
                num_classes = 2
                logits = torch.rand(batch_size, seq_len, num_classes)
                return MockOutput(logits)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MockModel().to(device)
        
        training_args = TrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            per_device_eval_batch_size=1,
            logging_dir="./test_logging",
            logging_steps=10,
            save_steps=0,
            save_total_limit=2,
            evaluation_strategy="no",
            remove_unused_columns=False
        )
        
        trainer = MultiLossExperimentalTrainer(
            device=device,
            model=model,
            args=training_args,
            sequential_loss_discount_factor=.99,
            class_weights=[.2, .8],
            max_length=10
        )
        
        # test Trainer initialization
        self.assertTrue(torch.allclose(trainer.class_weights, torch.tensor([.2, .8])))
        self.assertEqual(trainer.sequential_loss_discount_factor, .99)
        self.assertTrue(torch.allclose(
            trainer.discount_weights,
            torch.tensor([1.0, 0.99, 0.9801, 0.970299, 0.96059601,
                          0.9509900498999999, 0.941480149401, 0.9320653479069899, 0.9227446944279201, 0.9135172474836408])))
        self.assertEqual(trainer.class_imbalance_loss_weight, .5)
        self.assertEqual(trainer.sequential_loss_weight, .5)

        input_ids = torch.randint(0, 3, (2, 10, 5)).to(device)
        labels = torch.randint(0, 2, (2, 10, 1)).to(device)
        inputs = {"input_ids": input_ids, "labels": labels}

        loss, _ = trainer.compute_loss(model, inputs, return_outputs=True)
        self.assertIsNotNone(loss)

        # test that the class weights are applied correctly
        labels = [[0, 1], [1, 0]]
        labels = torch.tensor(labels).to(device)
        class_imbalance_weights = np.array([[5.  , 1.25], [1.25, 5.  ]])
        output_of_function = trainer.compute_class_imbalance_weights(labels)
        np.testing.assert_array_almost_equal(output_of_function, class_imbalance_weights)

    def test_time_to_prediction_positive(self):
        # Test case where the probabilities exceed the threshold at index 0
        probs = torch.tensor([
            [0.1, 0.9], 
            [0.2, 0.8], 
            [0.7, 0.3], 
            [0.8, 0.2], 
            [0.6, 0.4]
        ])
        threshold = 0.75
        assert evaluation.prediction_time_from_end_positive(probs, threshold) == 5

        # Test case where the probabilities exceeds the threshold at index 3
        probs = torch.tensor([
            [0.1, 0.1], 
            [0.2, 0.2], 
            [0.3, 0.3], 
            [0.4, 0.8], 
            [0.5, 0.5]
        ])
        assert evaluation.prediction_time_from_end_positive(probs, threshold) == 2
        print("time_to_prediction_positive() passed")

    def test_draw_probabilities_seqseq_to_seqlab(self):

        class MockOutput:
            def __init__(self, logits, true_labels):
                self.predictions = logits
                self.label_ids = true_labels

        input = MockOutput(
            logits=np.array(
                    [[[ 6.21448243e-01, 5.71545005e-01],
                    [ 2.14750147e-01, 6.02116823e-01],
                    [ 1.12612247e-02, 5.98399162e-01]],
                    [[ 7.21448243e-01, -5.71545005e-01],
                    [ 6.14750147e-01, -6.02116823e-01],
                    [ -6.12612247e-01, 9.98399162e0]]]),
            true_labels=np.array([[[0], [0], [0]],
                                      [[1], [1], [1]]])
        )
        expected_output = [0, 1], [0, 1]
        output = evaluation.draw_probabilities_seqseq_to_seqlab(input)
        np.testing.assert_array_equal(output, expected_output)

        input = MockOutput(
        logits=torch.tensor(
                [[[ 6.21448243e-01, 5.71545005e-01],
                [ 2.14750147e-01, 6.02116823e-01],
                [ 1.12612247e-02, 5.98399162e-01]],
                [[ 7.21448243e-01, -5.71545005e-01],
                [ 6.14750147e-01, -6.02116823e-01],
                [ -6.12612247e-01, 9.98399162e0]]]),
        true_labels=torch.tensor([[[0], [0], [0]],
                                    [[1], [-10], [-100]]])
        )
        expected_output = [0, 1], [0, 1]
        output = evaluation.draw_probabilities_seqseq_to_seqlab(input)
        np.testing.assert_array_equal(output, expected_output)


    def test_normalize_by_sequence_length(self):
    
        # Assume we have 3 sequences of lengths 2, 4, and 6, respectively
        mask = torch.tensor([[1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0],
                            [1, 1, 1, 1, 1, 1]])

        # Assume we have computed losses for each non-padded element
        losses = torch.tensor([1., 2., 3., 4., 1., 2., 3., 4., 1., 2., 3., 4.])

        # Compute the normalized loss
        normalized_loss = evaluation.normalize_by_sequence_length(losses=losses, mask=mask)

        expected_normalized_loss = np.mean(
            (np.mean((1, 2)),
             np.mean((3, 4, 1, 2)),
             np.mean((3, 4, 1, 2, 3, 4))))

        # Check that the computed normalized loss is close to the expected value
        np.testing.assert_almost_equal(normalized_loss.item(), expected_normalized_loss.item(), decimal=7)

if __name__ == "__main__":
    # test_time_to_prediction_positive()
    unittest.main()
