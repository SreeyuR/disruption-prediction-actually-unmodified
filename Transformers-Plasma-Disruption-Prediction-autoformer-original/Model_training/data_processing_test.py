import unittest
import torch
from data_processing import get_train_test_indices_from_Jinxiang_cases, get_class_weights_seq_to_seq 
import data_processing
import numpy as np

def _create_mock_dataset():
    mock_dataset = {0:{
            "label": 0,
            "data" : torch.randint(0, 3, (30, 5)),
            "machine": "d3d"
        },
        1:{
            "label": 1,
            "data" : torch.randint(0, 3, (30, 5)),
            "machine": "d3d"
        },
        2:{
            "label": 0,
            "data" : torch.randint(0, 3, (30, 5)),
            "machine": "d3d"
        },
        }
    return mock_dataset


def _dataset_to_MockReady(mock1):
    mockModelReady1 = data_processing.ModelReadyDatasetSeqtoSeqDisruption(
                shots=[mock1[i] for i in mock1.keys()],
                end_cutoff_timesteps=2,
                max_length=35,
                tau=5)
    return mockModelReady1


class TestDataProcessing(unittest.TestCase):

    def test_get_class_weights_seq_to_seq(self):
        # Create a mock dataset with known labels
        mock_dataset = [
            {"labels": torch.tensor([0, 0, 0, 1])},
            {"labels": torch.tensor([1, 1, 0, 0])},
            {"labels": torch.tensor([0, 0, 1, 1])},
            # Add more data as needed...
        ]

        # Expected class weights, calculated manually
        expected_class_weights = [7/12, 5/12]

        # Calculate class weights using the function
        class_weights = get_class_weights_seq_to_seq(mock_dataset)

        # Assert that the calculated class weights are close to the expected values
        for calculated_weight, expected_weight in zip(class_weights, expected_class_weights):
            self.assertAlmostEqual(calculated_weight, expected_weight, places=5)


    def test_ModelReadyDatasetSeqtoSeq(self):
        # Create a mock dataset with known labels
        mock1 = _create_mock_dataset()
        mockModelReady1 = _dataset_to_MockReady(mock1)
        
        # check length
        self.assertEqual(len(mockModelReady1), 3)

        # check end cutoff
        self.assertEqual(mockModelReady1[1]["labels"].size()[0], 28)

        # check tau windowing
        expected_label = torch.tensor(
            [0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 0, 0,
             0, 0, 0, 1, 1,
             1, 1, 1], dtype=torch.long).view(-1, 1)
        self.assertTrue(torch.allclose(mockModelReady1[1]["labels"], expected_label))

    def test_ModelReadySubset(self):
        mock1 = _create_mock_dataset()
        mockModelReady1 = _dataset_to_MockReady(mock1)
        
        # test subset
        mockSubset = mockModelReady1.subset([0, 1])
        self.assertEqual(len(mockSubset), 2)
        np.testing.assert_array_equal(mockSubset[0]["labels"], mockModelReady1[0]["labels"])


    def test_ModelReadySubsetAndConcat(self):
        mock1 = _create_mock_dataset()
        mock2 = _create_mock_dataset()

        mockModelReady1 = _dataset_to_MockReady(mock1)
        mockModelReady2 = _dataset_to_MockReady(mock2)
        
        # test concat
        combined = mockModelReady1.concat(mockModelReady2)
        self.assertEqual(len(combined), 6)
        np.testing.assert_array_equal(combined[0]["labels"].numpy(), mockModelReady1[0]["labels"].numpy())
        np.testing.assert_array_equal(combined[3]["labels"].numpy(), mockModelReady2[0]["labels"].numpy())

    def test_data_augmentation_seq_to_seq(self):
            mock1 = _create_mock_dataset()
            mockModelReady1 = _dataset_to_MockReady(mock1)
            
            mockModelReady1_augmented = data_processing.augment_training_set(
                mockModelReady1,
                seq_to_seq=True,
                balance_positive_and_negative_cases=True,
                ratio_to_augment=2
            )

            self.assertEqual(len(mockModelReady1_augmented), 6)
            
            # Assert that the arrays are not equal
            with np.testing.assert_raises(AssertionError):
                np.testing.assert_array_equal(mockModelReady1_augmented[0]["labels"], mockModelReady1[0]["labels"])
                np.testing.assert_array_equal(mockModelReady1_augmented[3]["labels"], mockModelReady1[0]["labels"])

            # Assert that there are three disruptions and three non-disruptions
            disruptions = 0
            non_disruptions = 0
            for i in range(len(mockModelReady1_augmented)):
                if mockModelReady1_augmented[i]["labels"][-1] == 0:
                    non_disruptions += 1
                else:
                    disruptions += 1

            np.testing.assert_equal(disruptions, 4)
            np.testing.assert_equal(non_disruptions, 2)


if __name__ == "__main__":
    unittest.main()