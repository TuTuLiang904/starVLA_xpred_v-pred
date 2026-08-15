import unittest

import torch

from fastwam.models.wan22.schedulers.scheduler_continuous import (
    WanContinuousFlowMatchScheduler,
    action_prediction_loss_token,
    action_prediction_step,
    normalize_action_prediction_type,
)


class TestActionPredictionModes(unittest.TestCase):
    def setUp(self):
        self.clean = torch.tensor([[[1.0, -2.0]], [[-3.0, 4.0]]])
        self.noise = torch.tensor([[[5.0, 6.0]], [[7.0, -8.0]]])
        self.timestep = torch.tensor([200.0, 800.0])
        sigma = self.timestep.view(-1, 1, 1) / 1000.0
        self.noisy = (1 - sigma) * self.clean + sigma * self.noise
        self.velocity = self.noise - self.clean

    def test_normalize_aliases(self):
        self.assertEqual(normalize_action_prediction_type("v_pred"), "v_prediction")
        self.assertEqual(normalize_action_prediction_type("x_pred"), "x_prediction")
        self.assertEqual(normalize_action_prediction_type("abot"), "x_prediction_v_loss")
        with self.assertRaisesRegex(ValueError, "Unsupported action_prediction_type"):
            normalize_action_prediction_type("epsilon")

    def test_perfect_predictions_have_zero_loss(self):
        cases = (
            ("v_prediction", self.velocity),
            ("x_prediction", self.clean),
            ("x_prediction_v_loss", self.clean),
        )
        for prediction_type, prediction in cases:
            loss = action_prediction_loss_token(
                prediction=prediction,
                clean_action=self.clean,
                noisy_action=self.noisy,
                velocity_target=self.velocity,
                timestep=self.timestep,
                num_train_timesteps=1000,
                prediction_type=prediction_type,
                x_pred_v_loss_eps=0.05,
            )
            torch.testing.assert_close(loss, torch.zeros_like(loss))

    def test_x_pred_v_loss_is_velocity_space_mse(self):
        prediction = self.clean + 0.5
        loss = action_prediction_loss_token(
            prediction=prediction,
            clean_action=self.clean,
            noisy_action=self.noisy,
            velocity_target=self.velocity,
            timestep=self.timestep,
            num_train_timesteps=1000,
            prediction_type="x_prediction_v_loss",
            x_pred_v_loss_eps=0.05,
        )
        expected = torch.tensor([[0.5**2 / 0.2**2], [0.5**2 / 0.8**2]])
        torch.testing.assert_close(loss, expected)

    def test_x_prediction_sampling_reaches_clean_action(self):
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        timestep, delta = scheduler.build_inference_schedule(
            num_inference_steps=4,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        for prediction_type in ("x_prediction", "x_prediction_v_loss"):
            sample = self.noise[:1].clone()
            clean = self.clean[:1]
            for step_t, step_delta in zip(timestep, delta):
                sample = action_prediction_step(
                    sample=sample,
                    prediction=clean,
                    timestep=step_t.unsqueeze(0),
                    delta=step_delta,
                    num_train_timesteps=scheduler.num_train_timesteps,
                    prediction_type=prediction_type,
                )
            torch.testing.assert_close(sample, clean)

    def test_v_prediction_step_matches_original_scheduler(self):
        scheduler = WanContinuousFlowMatchScheduler(num_train_timesteps=1000, shift=5.0)
        sample = self.noisy[:1]
        timestep = torch.tensor([500.0])
        delta = torch.tensor(-0.1)
        prediction = self.velocity[:1]
        actual = action_prediction_step(
            sample=sample,
            prediction=prediction,
            timestep=timestep,
            delta=delta,
            num_train_timesteps=scheduler.num_train_timesteps,
            prediction_type="v_prediction",
        )
        expected = scheduler.step(prediction, delta, sample)
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
