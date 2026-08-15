import unittest
import torch

from starVLA.model.modules.action_model.GR00T_ActionHeader import (
    flow_matching_step,
    flow_matching_target,
    normalize_prediction_type,
)


class TestGR00TPredictionType(unittest.TestCase):
    def test_prediction_type_aliases_and_validation(self):
        self.assertEqual(normalize_prediction_type("v_prediction"), "v_prediction")
        self.assertEqual(normalize_prediction_type("velocity"), "v_prediction")
        self.assertEqual(normalize_prediction_type("x_prediction"), "x_prediction")
        self.assertEqual(normalize_prediction_type("sample"), "x_prediction")
        with self.assertRaisesRegex(ValueError, "Unsupported prediction_type"):
            normalize_prediction_type("epsilon")

    def test_targets_match_gr00t_flow_definition(self):
        actions = torch.tensor([[[2.0, -1.0]]])
        noise = torch.tensor([[[-3.0, 4.0]]])
        self.assertTrue(torch.equal(flow_matching_target(actions, noise, "v_prediction"), actions - noise))
        self.assertTrue(torch.equal(flow_matching_target(actions, noise, "x_prediction"), actions))

    def test_perfect_predictions_reach_clean_action(self):
        noise = torch.tensor([[[2.0, -3.0]]])
        clean_action = torch.tensor([[[-1.0, 5.0]]])
        num_steps = 4

        for prediction_type in ("v_prediction", "x_prediction"):
            trajectory = noise.clone()
            for step in range(num_steps):
                time = step / num_steps
                next_time = (step + 1) / num_steps
                model_output = clean_action - noise if prediction_type == "v_prediction" else clean_action
                trajectory = flow_matching_step(trajectory, model_output, time, next_time, prediction_type)

            torch.testing.assert_close(trajectory, clean_action)
