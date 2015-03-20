package zx.soft.ann.jlmd.entity;

import zx.soft.ann.jlmd.Analyzer;
import zx.soft.ann.jlmd.parser.result.IResultParser;

/**
 * Entity to save obtained result
 */
@SuppressWarnings("rawtypes")
public class Result {

	private float successPercentage;
	private float quadraticError;
	private Analyzer analyzer;
	private IResultParser resultParser;

	public Result(Analyzer analyzer, IResultParser resultParser, float successPercentage, float quadraticError) {
		this.analyzer = analyzer;
		this.successPercentage = successPercentage;
		this.quadraticError = quadraticError;
		this.resultParser = resultParser;
	}

	/**
	 * Predict a value result from a array of elements
	 * @param element Array input to predict its output
	 * @return predicted output value
	 */
	public int predictValue(float[] element) {
		return (Integer) resultParser.parseResult(analyzer.getFOut(element));
	}

	public float getSuccessPercentage() {
		return successPercentage;
	}

	public float getQuadraticError() {
		return quadraticError;
	}

}
