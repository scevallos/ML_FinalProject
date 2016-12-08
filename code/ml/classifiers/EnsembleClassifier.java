package ml.classifiers;

import java.util.ArrayList;
import java.util.HashMap;

import ml.data.CrossValidationSet;
import ml.data.DataSet;
import ml.data.DataSetSplit;
import ml.data.Example;
import ml.utils.HashMapCounter;

/**
 * This classifier is used to implement the ensemble learning techniques:
 * boosting and bagging. The 4 classifiers used are Decision Tree, K-NN,
 * Perceptron, and Neural Nets.
 * 
 * @author SC, MM, AR, EZ
 *
 */
public class EnsembleClassifier implements Classifier {

	// Int constants used to ID the classifiers
	public static final int DT = 0;
	public static final int KNN = 1;
	public static final int PERC = 2;
	public static final int NN = 3;

	// HashMap holding the classifiers to be used
	private HashMap<Integer, Classifier> classifiers = new HashMap<Integer, Classifier>();

	// Contains the "scores" of each of the classifiers
	private HashMap<Integer, Double> alphas = new HashMap<Integer, Double>();

	/**
	 * Ensemble classifier; just a combination of the classifiers above
	 * 
	 * @param classifiersToUse
	 */
	public EnsembleClassifier(int[] classifiersToUse) {

		// Loop through the input array, seeing which classifiers to use
		for (int i = 0; i < classifiersToUse.length; i++) {
			switch (classifiersToUse[i]) {
			case 0:
				DecisionTreeClassifier dt = new DecisionTreeClassifier();
				dt.setDepthLimit(1);
				classifiers.put(0, dt);
				break;
			case 1:
				KNNClassifier knn = new KNNClassifier();
				classifiers.put(1, knn);
				break;
			case 2:
				AveragePerceptronClassifier perc = new AveragePerceptronClassifier();
				classifiers.put(2, perc);
				break;
			case 3:
				TwoLayerNN nn = new TwoLayerNN(2);
				classifiers.put(3, nn);
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
		int a = 0;
		// Trains each classifier on its respective data set
		for (Integer i : classifiers.keySet()) {
			classifiers.get(i).train(sets.get(a));
			a++;
		}
	}

	public double classify(Example example) {
		// Used to keep count of
		HashMapCounter<Double> predCount = new HashMapCounter<Double>();

		for (Integer i : classifiers.keySet())
			predCount.increment(classifiers.get(i).classify(example));

		// Sort them from most to least occurrences, and get the most occurring
		// TODO: If evenly split, randomly get the first or get most confident?
		double pred = predCount.sortedEntrySet().get(0).getKey();
		return pred;
	}

	/**
	 * 
	 */
	public void adaTrain(DataSet data) {
		// sets all the weights to 1/size of data
		setWeights(data, 1.0 / data.getData().size());

		for (Integer i : classifiers.keySet()) {
			Classifier c = classifiers.get(i);

			c.train(data);

			double alpha = calculateAlpha(data, c);

			double Z = 0.0;
			for (Example e : data.getData())
				Z += e.getWeight()
						* Math.exp(-alpha * e.getLabel() * c.classify(e));
			alphas.put(i, alpha);

			for (Example e : data.getData()) {

				double temp = (1 / Z) * e.getWeight()
						* Math.exp(-alpha * e.getLabel() * c.classify(e));
				e.setWeight(temp);
			}

		}

	}

	public double adaClassify(Example e) {
		double sum = 0;

		for (Integer i : classifiers.keySet())
			sum += alphas.get(i) * classifiers.get(i).classify(e);

		double pred = Math.signum(sum);
		return pred;
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
		double alpha = (.5 * Math.log((1 - epsilon) / epsilon));

		return alpha;
	}

	public static void main(String[] args) {
		String path = "/home/scevallos/Documents/ML/finalProject/data/titanic-train.perc.csv";
//		String path = "/home/scevallos/Documents/ML/finalProject/data/heart.csv";
		DataSet data = new DataSet(path, DataSet.CSVFILE);

		ArrayList<int[]> a = new ArrayList<int[]>();
		int[] four = { EnsembleClassifier.DT, EnsembleClassifier.NN,
				EnsembleClassifier.KNN, EnsembleClassifier.PERC };
		// int[] three1 = { EnsembleClassifier.DT, EnsembleClassifier.NN,
		// EnsembleClassifier.KNN };
		// int[] three2 = { EnsembleClassifier.DT, EnsembleClassifier.NN,
		// EnsembleClassifier.PERC };
		// int[] three3 = { EnsembleClassifier.DT, EnsembleClassifier.KNN,
		// EnsembleClassifier.PERC };
		// int[] three4 = { EnsembleClassifier.NN, EnsembleClassifier.KNN,
		// EnsembleClassifier.PERC };
		// int[] two1 = { EnsembleClassifier.NN, EnsembleClassifier.KNN };
		// int[] two12 = { EnsembleClassifier.KNN, EnsembleClassifier.NN };
		// int[] two2 = { EnsembleClassifier.NN, EnsembleClassifier.PERC };
		// int[] two3 = { EnsembleClassifier.NN, EnsembleClassifier.DT };
		// int[] two4 = { EnsembleClassifier.KNN, EnsembleClassifier.DT };
		// int[] two5 = { EnsembleClassifier.KNN, EnsembleClassifier.PERC };
		// int[] two6 = { EnsembleClassifier.PERC, EnsembleClassifier.DT };
//		 int[] knn = { EnsembleClassifier.KNN };
//		 int[] nn = { EnsembleClassifier.NN };
//		 int[] dt = { EnsembleClassifier.DT };
//		 int[] perc = { EnsembleClassifier.PERC };

		a.add(four);
		// a.add(three1);
		// a.add(three2);
		// a.add(three3);
		// a.add(three4);
		// a.add(two1);
		// a.add(two12);
		// a.add(two2);
		// a.add(two3);
		// a.add(two4);
		// a.add(two5);
		// a.add(two6);
//		 a.add(knn);
//		 a.add(nn);
//		 a.add(dt);
//		 a.add(perc);

//		int[] stumps = { EnsembleClassifier.DT, EnsembleClassifier.DT,
//				EnsembleClassifier.DT, EnsembleClassifier.DT };
//		a.add(stumps);

		for (int k = 0; k < a.size(); k++) {
			EnsembleClassifier ec = new EnsembleClassifier(a.get(k));
			for (int j = 0; j < a.get(k).length; j++)
				System.out.println("array is: " + a.get(k)[j]);

			// THIS IS JUST SPLITTING THE DATA once and classifying. Pretty
			// good:
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
			CrossValidationSet cv = new CrossValidationSet(data, 10, true);

			ArrayList<Double> percents = new ArrayList<Double>();
			long trainSum = 0;
			long classifySum = 0;
			for (int i = 0; i < 10; i++) {
				// System.out.println("Split Num: " + i);
				DataSetSplit split = cv.getValidationSet(i);

				// System.out.println("Training..");
				System.gc();
				long start = System.currentTimeMillis();
				ec.train(split.getTrain());
				trainSum += System.currentTimeMillis() - start;

				// System.out.println("Classifying");
				System.gc();
				start = System.currentTimeMillis();
				double correct = 0.0;
				for (Example e : split.getTest().getData()) {
					double pred = ec.classify(e);
					if (pred == e.getLabel())
						correct++;
				}
				classifySum += System.currentTimeMillis() - start;

				double percent = correct / split.getTest().getData().size();
				percents.add(percent);
				System.out.println("percent correct: " + percent);
			}
			// System.out.println("Average train time: " + ((double) trainSum)
			// / 10 / 1000 + "s");
			// System.out.println("Average test time: " + ((double) classifySum)
			// / 10 / 1000 + "s");

			double sum = 0.0;
			for (double p : percents)
				sum += p;

			double average = sum / 10.0;
			System.out.println("Average is: " + average);
		}
		
		
//		// Testing classifiers alone
//		KNNClassifier KNN = new KNNClassifier();
//		TwoLayerNN NN = new TwoLayerNN(2);
//		DecisionTreeClassifier DT = new DecisionTreeClassifier();
//		PerceptronClassifier PERC = new PerceptronClassifier();
//		
//		CrossValidationSet cv = data.getCrossValidationSet(10);
//		ArrayList<HashMap<String, Double>> percs = new ArrayList<HashMap<String, Double>>(10);
//		for(int c = 0; c < 10; c++){
//			DataSetSplit split = cv.getValidationSet(c);
//			
//			KNN.train(split.getTrain());
//			NN.train(split.getTrain());
//			DT.train(split.getTrain());
//			PERC.train(split.getTrain());
//			
//			double[] preds = {0.0, 0.0, 0.0, 0.0};
//			double[] cors = {0.0, 0.0, 0.0, 0.0};
//			for (Example e: split.getTest().getData()){
//				preds[0] = KNN.classify(e);
//				preds[1] = NN.classify(e);
//				preds[2] = DT.classify(e);
//				preds[3] = PERC.classify(e);
//				for(int j = 0; j < 4; j++){
//					if (preds[j] == e.getLabel())
//						cors[j]++;
//				}
//			}
//			
//			HashMap<String, Double> percents = new HashMap<String, Double>();
//			percents.put("KNN", cors[0]/split.getTest().getData().size());
//			percents.put("NN", cors[1]/split.getTest().getData().size());
//			percents.put("DT", cors[2]/split.getTest().getData().size());
//			percents.put("PERC", cors[3]/split.getTest().getData().size());
//			percs.add(percents);
//		}
//		System.out.println("percents are: " + percs);
//		System.out.println("done! :)");
	}
}
