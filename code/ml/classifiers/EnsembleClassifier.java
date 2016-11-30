package ml.classifiers;

import java.util.ArrayList;

import ml.data.DataSet;
import ml.data.Example;

public class EnsembleClassifier implements Classifier {
	
	public static final int DT = 0;
	public static final int KNN = 1;
	
	// ArrayList holding the classifiers to be used
	private ArrayList<Classifier> classifiers;
	
	public EnsembleClassifier(int[] classifiersToUse){
		// Initialize the ArrayList to whatever size it needs
		classifiers = new ArrayList<Classifier>(classifiersToUse.length);
		
		// Loop through the input array, seeing which classifiers to use
		for(int i = 0; i < classifiersToUse.length; i++){
			switch (classifiersToUse[i]){
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

	@Override
	public void train(DataSet data) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public double classify(Example example) {
		// TODO Auto-generated method stub
		return 0;
	}

	@Override
	public double confidence(Example example) {
		// TODO Auto-generated method stub
		return 0;
	}

}
