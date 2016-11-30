package ml.classifiers;

/**
 * Exception to throw when user attempts to set parameter to something outside
 * of specified range
 * 
 * @author MM & SC
 *
 */
public class InvalidParameterException extends Exception {

	/**
	 * Empty constructor
	 */
	public InvalidParameterException() {

	}

	public InvalidParameterException(String message) {
		super(message);
	}

}
