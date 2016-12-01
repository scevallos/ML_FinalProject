package ml.classifiers;

import java.util.ArrayList;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounter;

public class EnsembleClassifier implements Classifier {

	public static final int DT = 0;
	public static final int KNN = 1;
	public static final int PERC = 2;
	public static final int NN = 3;

	// ArrayList holding the classifiers to be used
	private ArrayList<Classifier> classifiers;

	ArrayList<Double> alphas = new ArrayList<Double>();

	/**
	 * Ensemble classifier
	 * 
	 * @param classifiersToUse
	 */
	public EnsembleClassifier(int[] classifiersToUse) {

		// Initialize the ArrayList to whatever size it needs
		classifiers = new ArrayList<Classifier>(classifiersToUse.length);

		// Loop through the input array, seeing which classifiers to use
		for (int i = 0; i < classifiersToUse.length; i++) {
			switch (classifiersToUse[i]) {
			case 0:
				DecisionTreeClassifier dt = new DecisionTreeClassifier();
				classifiers.add(dt);
				break;
			case 1:
				KNNClassifier knn = new KNNClassifier();
				classifiers.add(knn);
				break;
			case 2:
				AveragePerceptronClassifier perc = new AveragePerceptronClassifier();
				classifiers.add(perc);
				break;
			case 3:
				TwoLayerNN nn = new TwoLayerNN(2);
				classifiers.add(nn);
				break;
			}
		}
	}

	/**
	 * (BAGGING) Trains each classifier being used via its respective train
	 * method on its respective data set
	 * 
	 * @param data
	 */
	public void train(DataSet data) {
		// Gets the new data sets via bagging method
		ArrayList<DataSet> sets = data.getNewSets(classifiers.size());

		// Trains each classifier on its respective data set
		for (int i = 0; i < classifiers.size(); i++)
			classifiers.get(i).train(sets.get(i));

	}

	public double classify(Example example) {
		// Used to keep count of
		HashMapCounter<Double> predCount = new HashMapCounter<Double>();

		for (Classifier c : classifiers)
			predCount.increment(c.classify(example));

		// Sort them from most to least occurrences, and get the most occurring
		// TODO: If evenly split, randomly get the first or get most confident?
		return predCount.sortedEntrySet().get(0).getKey();
	}

	/**
	 * 
	 */
	public void adaTrain(DataSet data) {
		// sets all the weights to 1/size of data
		setWeights(data, 1 / data.getData().size());

		for (Classifier c : classifiers) {
			c.train(data);

			double alpha = calculateAlpha(data, c);

			double Z = 0.0;
			for (Example e : data.getData())
				Z += e.getWeight()
						* Math.exp(-alpha * e.getLabel() * c.classify(e));
			alphas.add(alpha);

			for (Example e : data.getData()) {

				double temp = (1 / Z) * e.getWeight()
						* Math.exp(-alpha * e.getLabel() * c.classify(e));
				e.setWeight(temp);
			}

		}

	}

	public double adaClassify(Example e) {
		int count = 0;
		long sum = 0;
		for (Double a : alphas) {
			System.out.println("sum: " + sum);
			sum += a * classifiers.get(count).classify(e);
			count++;
		}

		return Math.signum(sum);
	}

	@Override
	public double confidence(Example example) {
		// TODO Confidence of which classifier? Any of the ones that predicted
		// the maj label?
		return 0;
	}

	private void setWeights(DataSet data, double amount) {
		for (Example e : data.getData())
			e.setWeight(amount);

	}

	private double calculateAlpha(DataSet data, Classifier c) {
		double epsilon = 0;

		for (Example e : data.getData()) {
			double classify = c.classify(e);
			if (classify != e.getLabel())
				epsilon += e.getWeight();
		}

		return (.5 * Math.log((1 - epsilon) / epsilon));
	}

	public static void main(String[] args) {
		String path = "/home/scevallos/Documents/ML/finalProject/data/titanic-train.perc.csv";
		DataSet data = new DataSet(path, DataSet.CSVFILE);

		int[] classifiers = { EnsembleClassifier.KNN, EnsembleClassifier.PERC,
				EnsembleClassifier.NN, EnsembleClassifier.DT };
		EnsembleClassifier ec = new EnsembleClassifier(classifiers);

		// THIS IS JUST SPLITTING THE DATA once and classifying. Pretty good:
		// ~75%
		// DataSetSplit split = data.split(0.8);
		//
		// System.out.println("Training...");
		// ec.adaTrain(split.getTrain());
		// System.out.println("Done training!");
		//
		// System.out.println("Classifying...");
		// double correct = 0.0;
		// for (Example e : split.getTest().getData()) {
		// double pred = ec.adaClassify(e);
		// // System.out.println("pred: " + pred);
		// // System.out.println("label: " + e.getLabel());
		// if (pred == e.getLabel())
		// correct++;
		// }
		// double percent = correct / split.getTest().getData().size();
		// System.out.println("Percent Correct: " + percent);

		// CROSS VALIDATION SET TESTING: Averages a 70.6%
		CrossValidationSet cv = data.getCrossValidationSet(10);
		for (int i = 0; i < 10; i++) {
			System.out.println("Split Num: " + i);
			DataSetSplit split = cv.getValidationSet(i);

			System.out.println("Training..");
			ec.train(split.getTrain());
			System.out.println("Classifying");
			double correct = 0.0;
			for (Example e : split.getTest().getData()) {
				double pred = ec.classify(e);
				if (pred == e.getLabel())
					correct++;
			}

			double percent = correct / split.getTest().getData().size();
			System.out.println("percent correct: " + percent);
		}

	}

}
