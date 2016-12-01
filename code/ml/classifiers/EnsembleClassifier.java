package ml.classifiers;

import java.util.ArrayList;

import ml.data.DataSet;
import ml.data.Example;
import ml.utils.HashMapCounter;

public class EnsembleClassifier implements Classifier {

	public static final int DT = 0;
	public static final int KNN = 1;

	// ArrayList holding the classifiers to be used
	private ArrayList<Classifier> classifiers;

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

	@Override
	public double confidence(Example example) {
		// TODO Confidence of which classifier? Any of the ones that predicted
		// the maj label?
		return 0;
	}

	public static void main(String[] args) {
		HashMapCounter<Double> c = new HashMapCounter<Double>();
		c.increment(1.0);
		// c.increment(1.0);
		c.increment(0.0);
		// c.increment(0.0);
		// c.increment(1.0);
		// c.increment(1.0);
		// c.increment(0.0);
		// 1.0 : 4, 0.0 : 3
		System.out.println(c.sortedEntrySet());
		System.out.println(c.sortedEntrySet().get(0).getKey());
	}

}
