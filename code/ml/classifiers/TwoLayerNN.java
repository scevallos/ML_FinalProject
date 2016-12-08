package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Set;
import java.util.Collections;
import java.util.Random;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounterDouble;

/**
 * A class representing a neural network with two hidden layers.
 * 
 * @author Sebastian Cevallos & Maria Martinez
 *
 */
public class TwoLayerNN implements Classifier {

	private static boolean testing = false;

	// File locations
	private static final String TITANIC_DATA = "/Users/sebastiancevallos/Documents/workspace/ML_Assign3/titanic-train.perc.csv";
	private static final String TEST_DATA = "/Users/sebastiancevallos/Documents/workspace/Assign8/test.csv";

	// Store confidence after classification
	private double confidence;

	// Map of all input weights (feature index, counter of weight index ->
	// weight)
	private HashMap<Integer, HashMapCounterDouble<Integer>> inWeights;

	// Map of output weights (hidden node index --> weight)
	private HashMap<Integer, Double> outWeights;

	// h and v vectors
	private ArrayList<Double> h, v;

	// Generator used to get initial weight values
	private Random r = new Random();

	// Learning rate
	private double eta;

	// Number of hidden nodes
	private int hidNodes;

	// Number of times to iterate while training
	private int numIters;

	// The copy of the training data set with the added bias feature
	private DataSet biasCopy;

	/**
	 * Creates the TwoLayerNN classifier, sets default values of eta (0.1) and
	 * numIters (200).
	 * 
	 * @param hidNodes
	 *            number of hidden nodes to have in the hidden layer
	 */
	public TwoLayerNN(int hidNodes) {
		this.hidNodes = hidNodes;
		eta = 0.1;
		numIters = 200;
	}

	/**
	 * Get random values between -0.1 and 0.1 to initialize input weights with
	 * 
	 * @param features
	 *            the set of features to learn over
	 * @return HashMap of integers, referring to feature indices and map
	 *         counters (mapping the index of the node and its corresponding
	 *         weight)
	 */
	protected HashMap<Integer, HashMapCounterDouble<Integer>> getRandomInWeights(
			Set<Integer> features) {
		HashMap<Integer, HashMapCounterDouble<Integer>> temp = new HashMap<Integer, HashMapCounterDouble<Integer>>();

		for (Integer f : features) {
			HashMapCounterDouble<Integer> counter = new HashMapCounterDouble<Integer>();
			for (int i = 0; i < hidNodes; i++) {
				// Get random double between -0.1 and 0.1
				double val = -0.1 + 0.2 * r.nextDouble();
				assert (val <= 0.1 && val >= -0.1);
				counter.put(i, val);
			}
			temp.put(f, counter);
		}

		return temp;
	}

	/**
	 * Get random values between -0.1 and 0.1 to initialize output weights
	 * 
	 * 
	 * @return HashMap of ints (referring to the node) and doubles (the
	 *         corresponding weight)
	 */
	protected HashMap<Integer, Double> getRandomOutWeights() {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		// Getting random weights for hidden node outputs
		for (int i = 0; i < hidNodes; i++) {
			double val = -0.1 + 0.2 * r.nextDouble();
			assert (val <= 0.1 && val >= -0.1);
			temp.put(i, val);
		}

		// Getting random weight for bias
		double val = -0.1 + 0.2 * r.nextDouble();
		assert (val <= 0.1 && val >= -0.1);
		temp.put(hidNodes, val);

		return temp;
	}

	/**
	 * Initialize all the network weights
	 * 
	 * @param features
	 *            set of features to learn on
	 */
	protected void initializeWeights(Set<Integer> features) {
		inWeights = getRandomInWeights(features);
		outWeights = getRandomOutWeights();
	}

	// Used to initialize the weights to what the example in the write up has
	private void testWeights() {
		HashMap<Integer, HashMapCounterDouble<Integer>> in = new HashMap<Integer, HashMapCounterDouble<Integer>>();
		HashMap<Integer, Double> out = new HashMap<Integer, Double>();

		// feature 0 counter
		HashMapCounterDouble<Integer> counter0 = new HashMapCounterDouble<Integer>();
		counter0.put(0, -0.7);
		counter0.put(1, 0.03);

		// feature 1 counter
		HashMapCounterDouble<Integer> counter1 = new HashMapCounterDouble<Integer>();
		counter1.put(0, 1.6);
		counter1.put(1, 0.6);

		// feature 2 counter
		HashMapCounterDouble<Integer> counter2 = new HashMapCounterDouble<Integer>();
		counter2.put(0, -1.8);
		counter2.put(1, -1.4);

		// Putting into in weights map
		in.put(0, counter0);
		in.put(1, counter1);
		in.put(2, counter2);

		// Output weights
		out.put(0, -1.1);
		out.put(1, -0.6);
		out.put(2, 1.8);

		inWeights = in;
		outWeights = out;
	}

	@Override
	/**
	 * Trains on the given data set via the NN algorithm
	 * 
	 * @param data
	 * 
	 */
	public void train(DataSet data) {

		// Add a bias value to the training set
		DataSet biasCopy = data.getCopyWithBias();

		// Store the new data set
		this.biasCopy = biasCopy;

		// Initializing weights
		if (testing)
			testWeights();
		else
			initializeWeights(biasCopy.getAllFeatureIndices());

		// Get training set of examples
		ArrayList<Example> training = biasCopy.getData();

		for (int iter = 0; iter < numIters; iter++) {
			// Randomize data order
			Collections.shuffle(training);

			for (Example e : training) {
				// update each output weight

				// Make vector of hidden node outputs
				h = new ArrayList<Double>(hidNodes + 1);

				// Populate h vector
				for (int o = 0; o < hidNodes; o++) {
					h.add(Math.tanh(calcNodeInput(o, e)));
				}

				if (testing)
					System.out.println("h: " + h);

				// Add bias value to vector
				h.add(1.0);

				// Make vector of output weights from hidden nodes
				v = new ArrayList<Double>(hidNodes + 1);

				// Populate vector
				for (int vk = 0; vk < hidNodes + 1; vk++) {
					v.add(outWeights.get(vk));
				}

				double vDotH = dot(v, h);

				double vUpdate = 0.0;
				double deltaOut = (e.getLabel() - Math.tanh(vDotH))
						* (1 - Math.pow(Math.tanh(vDotH), 2));
				for (int p = 0; p < hidNodes + 1; p++) {
					vUpdate = eta * h.get(p) * deltaOut;
					outWeights.put(p, outWeights.get(p) + vUpdate);
				}

				// update each input weight
				for (int node = 0; node < hidNodes; node++) {
					for (int feat = 0; feat < e.getFeatureSet().size(); feat++) {
						double wDotX = calcNodeInput(node, e);

						double wUpdate = eta * e.getFeature(feat)
								* (1 - Math.pow(Math.tanh(wDotX), 2))
								* outWeights.get(node) * deltaOut;

						inWeights.get(feat).put(node,
								inWeights.get(feat).get(node) + wUpdate);
					}
				}
			}

			/*
			 * QUESTION 1 ACCURACY CODE (also had test data as a param)
			 * 
			 * System.out.println("computing test accuracy..."); // test
			 * accuracy double correctTest = 0.0; for (Example e :
			 * test.getData()) { double pred = classify(e); if (pred ==
			 * e.getLabel()) correctTest++; } double testAccuracy = correctTest
			 * / test.getData().size();
			 * 
			 * System.out.println("computing training accuracy...");
			 * 
			 * // training accuracy & squared error double squaredError = 0.0;
			 * double correctTrain = 0.0; for (Example e : training) { double
			 * pred = classify(e, true); if (pred == e.getLabel())
			 * correctTrain++; squaredError += Math.pow(e.getLabel() -
			 * confidence(e, true), 2); } double trainAccuracy = correctTrain /
			 * training.size();
			 * 
			 * System.out.println("ITER: " + iter); System.out.println(
			 * "TRAINING ACC: " + trainAccuracy); System.out.println(
			 * "TEST ACC: " + testAccuracy); System.out.println("SQUARED ERR: "
			 * + squaredError);
			 * 
			 * System.out.println("Done computing accuracies");
			 */
		}

	}

	/**
	 * Computes the input to the specified hidden node (dot prod of example with
	 * corresponding weight vector)
	 * 
	 * @param hiddenNodeIndex
	 * @param e
	 * @return
	 */
	private double calcNodeInput(int hiddenNodeIndex, Example e) {
		// Input vector for this example
		ArrayList<Double> x = new ArrayList<Double>();

		// Populate the vector with the input values & bias
		Set<Integer> featureSet = e.getFeatureSet();
		for (Integer feat : featureSet) {
			x.add(e.getFeature(feat));
		}

		int numFeatures = e.getFeatureSet().size();

		ArrayList<Double> w = new ArrayList<Double>(numFeatures);
		for (int n = 0; n < numFeatures; n++) {
			w.add(inWeights.get(n).get(hiddenNodeIndex));
		}

		return dot(x, w);

	}

	public double classify(Example e, boolean hasBias) {
		if (hasBias) {
			ArrayList<Double> h = new ArrayList<Double>(e.getFeatureSet()
					.size());

			for (int h_in = 0; h_in < hidNodes; h_in++) {
				h.add(calcNodeInput(h_in, e));
			}

			// adding bias
			h.add(1.0);

			confidence = Math.tanh(dot(h, v));

			if (confidence > 0)
				return 1.0;
			else
				return -1.0;
		} else
			return classify(e);

	}

	@Override
	/**
	 * Classifies the given example on the learned model
	 * 
	 * @param e
	 * @param hasBias
	 * @return
	 */
	public double classify(Example example) {

		// Add bias feature to the example
		Example e = biasCopy.addBiasFeature(example);

		ArrayList<Double> h = new ArrayList<Double>(e.getFeatureSet().size());

		for (int h_in = 0; h_in < hidNodes; h_in++) {
			h.add(calcNodeInput(h_in, e));
		}

		// adding bias
		h.add(1.0);

		confidence = Math.tanh(dot(h, v));

		if (confidence > 0)
			return 1.0;
		else
			return -1.0;

	}

	@Override
	/**
	 * Returns the absolute value of the output for classifying this example
	 * 
	 * @param example
	 *            to be classified
	 * @return
	 */
	public double confidence(Example example) {
		classify(example);

		return Math.abs(confidence);
	}

	/**
	 * Sets the learning rate to the specified double
	 * 
	 * @param eta
	 */
	public void setEta(double eta) {
		this.eta = eta;
	}

	/**
	 * Sets the number of times to iterate while training
	 * 
	 * @param numIters
	 */
	public void setIterations(int numIters) {
		this.numIters = numIters;
	}

	/**
	 * Helper function used to take the dot product of two vectors
	 * 
	 * @param x1
	 *            ArrayList of doubles (first vector)
	 * @param x2
	 *            ArrayList of doubles (second vector)
	 * @return
	 */
	private double dot(ArrayList<Double> x1, ArrayList<Double> x2) {
		assert x1.size() == x2.size();
		double ans = 0.0;
		for (int i = 0; i < x1.size(); i++) {
			ans += x1.get(i) * x2.get(i);
		}

		return ans;
	}

	/**
	 * Used for debugging; Prints out all the weights of the currently learned
	 * model
	 * 
	 */
	private void prettyPrintWeights() {
		System.out.println("Input Weights:");
		for (int f = 0; f < biasCopy.getAllFeatureIndices().size(); f++) {
			for (int n = 0; n < hidNodes + 1; n++) {
				System.out.print("feat " + f + ", node " + n + ": "
						+ inWeights.get(f).get(n) + "\n");
			}
		}
		System.out.println("Output Weights:");
		for (int o = 0; o < hidNodes + 1; o++) {
			System.out.print("v_" + o + ": " + outWeights.get(o) + "\n");
		}
	}

	public static void main(String[] args) {
		String path = "/home/scevallos/Documents/ML/finalProject/data/titanic-train.perc.csv";
		DataSet data = new DataSet(path, DataSet.CSVFILE);
		// if (testing) {
		// data = new DataSet(TEST_DATA, DataSet.CSVFILE);
		// } else {
		// data = new DataSet(TITANIC_DATA, DataSet.CSVFILE);
		// }

		CrossValidationSet CV = new CrossValidationSet(data, 10);
		int hiddenNodes = 2;
		TwoLayerNN nn = new TwoLayerNN(hiddenNodes);

		nn.setEta(0.5);
		nn.setIterations(1);

		for (int i = 0; i < 10; i++) {
			System.out.println("Split num: " + i);
			DataSetSplit split = CV.getValidationSet(0);
			System.out.println("Training...");
			nn.train(split.getTrain());
			System.out.println("Testing..");
			double correct = 0.0;
			for (Example e : split.getTest().getData()) {
				double pred = nn.classify(e);
				if (pred == e.getLabel())
					correct++;
			}
			double percent = correct / split.getTest().getData().size();
			System.out.println("percent: " + percent);
		}

		// DataSetSplit split = null;
		// System.out.println("Training...");
		// if (testing) {
		// nn.train(data);
		// } else {
		// ETA TESTING CODE
		// double etaVal = 1.0;
		// for (int et = 0; et < 25; et++) {
		// for (int i = 0; i < 10; i++) {
		// double correctTrain = 0.0;
		// double correctTest = 0.0;
		// int hiddenNodes = 2;
		// TwoLayerNN nn = new TwoLayerNN(hiddenNodes);
		//
		// nn.setEta(etaVal);
		//
		// split = CV.getValidationSet(i);
		// nn.train(split.getTrain());
		//
		// System.out.println("Classifying...");
		//
		// for (Example e : split.getTrain().getData()) {
		// double predTrain = nn.classify(e, true);
		// if (predTrain == e.getLabel())
		// correctTrain++;
		// }
		//
		// System.out.println("Train Accuracy: (etaVal: " + etaVal + ")
		// (splitNum: " + i + ") "
		// + correctTrain / split.getTrain().getData().size());
		//
		// for (Example e : split.getTest().getData()) {
		// double predTest = nn.classify(e);
		// if (predTest == e.getLabel())
		// correctTest++;
		// }
		//
		// System.out.println("Test Accuracy: (etaVal: " + etaVal + ")
		// (splitNum: " + i + ") "
		// + correctTest / split.getTest().getData().size());
		// }
		// etaVal *= 0.9;
		// }
		// }
		// System.out.println("Done training");

		// if (!testing) {
		// } else {
		// nn.prettyPrintWeights();
		//
		// }
		// for (int e = 0; e < split.getTest().getData().size(); e++) {
		// for (Example e : data.getData()) {
		// System.out.println("on example " + e + " out of " +
		// split.getTest().getData().size());
		// double pred = nn.classify(split.getTest().getData().get(e));
		// if (pred == split.getTest().getData().get(e).getLabel())
		// correct++;

		// System.out.println("Accuracy: " + correct / data.getData().size());

		// System.out.println("classifying...");
		// double pred = nn.classify(data.getData().get(0));
		// double conf = nn.confidence(data.getData().get(0));
		// System.out.println("pred is: " + pred);
		// System.out.println("conf is: " + conf);
	}

}
