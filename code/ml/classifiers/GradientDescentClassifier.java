package ml.classifiers;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Set;
import java.util.Random;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;

/**
 * Gradient descent classifier allowing for two different loss functions and
 * three different regularization settings.
 * 
 * @author Maria Martinez & Sebastian Cevallos
 *
 */
public class GradientDescentClassifier implements Classifier {
	// constants for the different surrogate loss functions
	public static final int EXPONENTIAL_LOSS = 0;
	public static final int HINGE_LOSS = 1;

	// constants for the different regularization parameters
	public static final int NO_REGULARIZATION = 0;
	public static final int L1_REGULARIZATION = 1;
	public static final int L2_REGULARIZATION = 2;

	protected HashMap<Integer, Double> weights; // the feature weights
	protected double b = 0; // the intersect weight

	protected int iterations = 10;

	// Parameters to keep track of
	private int loss;
	private int reg;
	private double lambda;
	private double eta;

	/**
	 * Zero parameter constructor simply sets defaults of the classifier
	 */
	public GradientDescentClassifier() {
		// Default behavior below
		loss = this.EXPONENTIAL_LOSS;
		reg = this.NO_REGULARIZATION;
		lambda = 0.01;
		eta = 0.01;
	}

	/**
	 * Takes an int and selects the loss function to use (based on the
	 * constants)
	 * 
	 * @param lossType
	 * @throws InvalidParameterException
	 */
	public void setLoss(int lossType) throws InvalidParameterException {
		// Check that it's valid first (between 0 & 1 inclusive)
		if (lossType > -1 && lossType < 2)
			loss = lossType;
		else
			throw new InvalidParameterException("Invalid loss type chosen!");
	}

	/**
	 * Takes an int and selects the regularization method to use
	 * 
	 * @param regType
	 * @throws InvalidParameterException
	 */
	public void setRegularization(int regType) throws InvalidParameterException {
		// Check that it's valid first (between 0 & 2 inclusive)
		if (regType > -1 && regType < 3)
			reg = regType;
		else
			throw new InvalidParameterException(
					"Invalid regularization type chosen!");

	}

	/**
	 * Takes a double and sets that as the new lambda to use.
	 * 
	 * @param lambda
	 */
	public void setLambda(double lambda) {
		this.lambda = lambda;
	}

	/**
	 * Takes a double and sets that as the new eta to use.
	 * 
	 * @param eta
	 */
	public void setEta(double eta) {
		this.eta = eta;
	}

	/**
	 * Trains the data set based on the Gradient Descent algorithm.
	 * 
	 * @param data
	 *            to be trained
	 */
	public void train(DataSet data) {
		initializeWeights(data.getAllFeatureIndices());

		ArrayList<Example> training = (ArrayList<Example>) data.getData()
				.clone();

		for (int it = 0; it < iterations; it++) {
			Collections.shuffle(training);

			// Iterate through all the examples
			for (Example e : training) {

				// Get the label and dot product of the weight vector and the
				// feat vector
				double label = e.getLabel();
				double dotProduct = this.getDistanceFromHyperplane(e, weights,
						b);

				double lossVal = 0.0;
				if (loss == this.EXPONENTIAL_LOSS) {
					lossVal = Math.exp((-label) * (dotProduct));
				} else
					// loss == this.HINGE_LOSS
					lossVal = Math.max(0.0, (1 - (label * dotProduct)));

				// Used to store the regularization term
				double r = 0.0;

				// Iterate on the features
				for (Integer featureIndex : e.getFeatureSet()) {
					double oldWeight = weights.get(featureIndex);
					double featureValue = e.getFeature(featureIndex);

					r = 0.0;
					// if no regularization, just leave as zero
					if (reg == this.L1_REGULARIZATION)
						r = lambda * Math.signum(oldWeight);

					else if (reg == this.L2_REGULARIZATION)
						r = lambda * oldWeight;

					weights.put(featureIndex, oldWeight + eta
							* (label * featureValue * lossVal - lambda * r));
				}
				// Set up regularizer for b
				double r2 = 0.0; // if no regularization, just leave as zero
				if (reg == this.L1_REGULARIZATION)
					r2 = lambda * Math.signum(b);
				else if (reg == this.L2_REGULARIZATION)
					r2 = lambda * b;

				// update b
				b += eta * (label * lossVal - lambda * r2);

			}
		}
	}

	/**
	 * Classifies the given example based on the trained model.
	 * 
	 * @param Example
	 *            to be classified
	 */
	public double classify(Example example) {
		return getPrediction(example);
	}

	/**
	 * Get a weight vector over the set of features with each weight set to 0
	 * 
	 * @param features
	 *            the set of features to learn over
	 * @return
	 */
	protected HashMap<Integer, Double> getZeroWeights(Set<Integer> features) {
		HashMap<Integer, Double> temp = new HashMap<Integer, Double>();

		for (Integer f : features) {
			temp.put(f, 0.0);
		}

		return temp;
	}

	/**
	 * Initialize the weights and the intersect value
	 * 
	 * @param features
	 */
	protected void initializeWeights(Set<Integer> features) {
		weights = getZeroWeights(features);
		b = 0;
	}

	/**
	 * Set the number of iterations the perceptron should run during training
	 * 
	 * @param iterations
	 */
	public void setIterations(int iterations) {
		this.iterations = iterations;
	}

	/**
	 * Returns the distance from the confidence of the prediction for the given
	 * exa.mple
	 * 
	 * @param Example
	 *            to get confidence of
	 */
	public double confidence(Example example) {
		return Math.abs(getDistanceFromHyperplane(example, weights, b));
	}

	/**
	 * Get the prediction from the current set of weights on this example
	 * 
	 * @param e
	 *            the example to predict
	 * @return
	 */
	protected double getPrediction(Example e) {
		return getPrediction(e, weights, b);
	}

	/**
	 * Get the prediction from the on this example from using weights w and
	 * inputB
	 * 
	 * @param e
	 *            example to predict
	 * @param w
	 *            the set of weights to use
	 * @param inputB
	 *            the b value to use
	 * @return the prediction
	 */
	protected static double getPrediction(Example e,
			HashMap<Integer, Double> w, double inputB) {
		double sum = getDistanceFromHyperplane(e, w, inputB);

		if (sum > 0) {
			return 1.0;
		} else if (sum < 0) {
			return -1.0;
		} else {
			return 0;
		}
	}

	/**
	 * Computes the distance from the hyperplane of the given example.
	 * 
	 * @param e
	 *            example to measure distance of
	 * @param w
	 *            weights hashmap
	 * @param inputB
	 *            b in the current model
	 * @return
	 */
	protected static double getDistanceFromHyperplane(Example e,
			HashMap<Integer, Double> w, double inputB) {
		double sum = inputB;

		// for(Integer featureIndex: w.keySet()){
		// only need to iterate over non-zero features
		for (Integer featureIndex : e.getFeatureSet()) {
			sum += w.get(featureIndex) * e.getFeature(featureIndex);
		}

		return sum;
	}

	/**
	 * Nice print of the classifier's weights
	 * 
	 */
	public String toString() {
		StringBuffer buffer = new StringBuffer();

		ArrayList<Integer> temp = new ArrayList<Integer>(weights.keySet());
		Collections.sort(temp);

		for (Integer index : temp) {
			buffer.append(index + ":" + weights.get(index) + " ");
		}

		return buffer.substring(0, buffer.length() - 1);
	}

	// Testing..
	public static void main(String[] args) throws InvalidParameterException {
		String titanic = "/home/scevallos/Documents/ML/titanic-train.perc.csv";

		DataSet data = new DataSet(titanic, DataSet.CSVFILE);

		CrossValidationSet cvSet = new CrossValidationSet(data, 10);

		// Copy-pasted in whatever was previously determined as the best
		// eta/lambda
		double lambda = 0.3487;
		double eta = 0.0382;

		// Prepare the Classifier
		GradientDescentClassifier GDC = new GradientDescentClassifier();
		GDC.setRegularization(NO_REGULARIZATION);
		GDC.setLoss(EXPONENTIAL_LOSS);

		ArrayList<Double> avgs = new ArrayList<Double>();
		ArrayList<Double> etas = new ArrayList<Double>();
		ArrayList<Double> lams = new ArrayList<Double>();

		ArrayList<Double> pcents = new ArrayList<Double>();

		for (int i = 0; i < 10; i++) {

			lams.add(lambda);
			etas.add(eta);

			GDC.setLambda(lambda);
			GDC.setEta(eta);

			double correct = 0.0;
			double avg = 0.0;
			for (int splitNum = 0; splitNum < 10; splitNum++) {
				correct = 0.0;
				DataSetSplit temp = cvSet.getValidationSet(splitNum);
				// set hyper-parameters
				GDC.train(temp.getTrain());

				for (Example e : temp.getTest().getData()) {
					double pred = GDC.classify(e);
					if (pred == e.getLabel())
						correct++;
				}
				pcents.add(correct / (temp.getTest().getData().size()));
				avg += correct / (temp.getTest().getData().size());
			}
			avgs.add(avg / 10);
			avg = 0.0;
			
			// Select one to vary 
			// lambda *= 0.9;
			// eta *= 0.9;

		}
		int maxIndex = 0;
		for (int i = 1; i < pcents.size(); i++) {
			if (pcents.get(i) > pcents.get(maxIndex))
				maxIndex = i;
		}

		System.out.println("Best score: " + pcents.get(maxIndex));
		System.out.println("All scores: " + pcents);
	}
}
