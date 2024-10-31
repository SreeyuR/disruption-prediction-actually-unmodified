import unittest
import torch
from data_processing import get_train_test_indices_from_Jinxiang_cases 
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


if __name__ == "__main__":
    unittest.main()