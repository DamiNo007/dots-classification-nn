import java.util.function.UnaryOperator;

public class NeuralNetwork {

    private double learningRate;
    private Layer[] layers;
    private UnaryOperator<Double> activation;
    private UnaryOperator<Double> derivative;

    public NeuralNetwork(double learningRate, UnaryOperator<Double> acitvation, UnaryOperator<Double> derivative, int... sizes) {
        this.learningRate = learningRate;
        this.activation = acitvation;
        this.derivative = derivative;
        layers = new Layer[sizes.length];

        for (int i = 0; i < sizes.length; i++) {
            int nextSize = 0;
            if (i < sizes.length - 1) nextSize = sizes[i + 1];
            layers[i] = new Layer(sizes[i], nextSize);
            for (int j = 0; j < sizes[i]; j++) {
                layers[i].biases[j] = Math.random() * 2.0 - 1.0;
                for (int k = 0; k < nextSize; k++) {
                    layers[i].weights[j][k] = Math.random() * 2.0 - 1.0;
                }
            }
        }
    }

    public double[] feedForward(double[] inputs) {
        System.arraycopy(inputs, 0, layers[0].neurons, 0, inputs.length);
        for (int i = 1; i < layers.length; i++) {
            Layer lPrev = layers[i - 1];
            Layer lCurr = layers[i];

            for (int j = 0; j < lCurr.size; j++) {
                lCurr.neurons[j] = 0;
                for (int k = 0; k < lPrev.size; k++) {
                    lCurr.neurons[j] += lPrev.neurons[k] * lPrev.weights[k][j];
                }
                lCurr.neurons[j] += lCurr.biases[j];
                lCurr.neurons[j] = activation.apply(lCurr.neurons[j]);
            }
        }
        return layers[layers.length - 1].neurons;
    }

    public void backpropagation(double[] targets) {
        double[] errors = new double[layers[layers.length - 1].size];
        for (int i = 0; i < layers[layers.length - 1].size; i++) {
            errors[i] = targets[i] - layers[layers.length - 1].neurons[i];
        }

        for (int k = layers.length - 1; k >= 1; k--) {
            Layer lPrev = layers[k - 1];
            Layer lCurr = layers[k];

            double[] errorsOfThePrevLayer = new double[lPrev.size];
            double[] currGradients = new double[lCurr.size];
            for (int i = 0; i < lCurr.size; i++) {
                currGradients[i] = learningRate * errors[i] * derivative.apply(lCurr.neurons[i]);
            }

            for (int i = 0; i < lCurr.size; i++) {
                for(int j =0; j < lPrev.size; j++){
                    lPrev.weights[j][i] += currGradients[i] * lPrev.neurons[j];
                }
            }

            for(int i = 0; i <lPrev.size; i++){
                errorsOfThePrevLayer[i] = 0 ;
                for(int j =0; j<lCurr.size; j++){
                    errorsOfThePrevLayer[i] += lPrev.weights[i][j]*errors[j];
                }
            }

            errors = new double[lPrev.size];
            System.arraycopy(errorsOfThePrevLayer, 0, errors, 0 ,lPrev.size);

            for(int i =0; i<lCurr.size; i++){
                lCurr.biases[i] += currGradients[i];
            }
        }
    }
}

