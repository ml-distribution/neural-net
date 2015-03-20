package zx.soft.ann.jlmd.transfer;

/**
 * Interface for transfer function. The function of this is limit the values of generated output
 */
public interface ITransferFunction {

	/**
	 * Calculate the transfer value limited by a function
	 * @param value Transfer value
	 * @return obtained value
	 */
	float transfer(float value);

}
