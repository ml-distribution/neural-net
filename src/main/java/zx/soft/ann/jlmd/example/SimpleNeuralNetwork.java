package zx.soft.ann.jlmd.example;

import zx.soft.ann.jlmd.NeuralNetwork;
import zx.soft.ann.jlmd.callback.INeuralNetworkCallback;
import zx.soft.ann.jlmd.entity.Error;
import zx.soft.ann.jlmd.entity.Result;
import zx.soft.ann.jlmd.utils.DataUtils;

public class SimpleNeuralNetwork {

	public static void main(String[] args) {

		System.out.println("Starting neural network sample... ");

		float[][] x = DataUtils.readInputsFromFile("data/x.txt");
		int[] t = DataUtils.readOutputsFromFile("data/t.txt");

		NeuralNetwork neuralNetwork = new NeuralNetwork(x, t, new INeuralNetworkCallback() {
			@Override
			public void success(Result result) {
				float[] valueToPredict = new float[] { -0.205f, 0.780f };
				System.out.println("Success percentage: " + result.getSuccessPercentage());
				System.out.println("Predicted result: " + result.predictValue(valueToPredict));
			}

			@Override
			public void failure(Error error) {
				System.out.println("Error: " + error.getDescription());
			}
		});

		neuralNetwork.startLearning();
	}

}
