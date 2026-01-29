The **pre\_processing** folder contains the python scripts that generate the pre processed data and stores it in .pt format.

It is stored in this format for faster redablity while traning the ML model.

There are different versions of pre processed data that is required for the training different ML models.

* Pre processed nodes and weights data 
* pre processed weight scales 
* pre processed scales and Centers

All the python script required for preprocessing the data and the data itself if available in the folder Pre\_processing.



The classic\_FNN and Scalling\_FNN are two ML model that predicts quadrature nodes and weights.



In both cases, a polynomial interface is given in symbolic form through its exponents and coefficients. This is first evaluated on a fixed Cartesian grid using a nodal preprocessor. The resulting nodal field is normalized and used as input to a neural network. The network then predicts quantities that define a quadrature rule.



**The two folders differ only in what the network is asked to predict.**



* The Classic\_FNN folder contains a model that directly predicts quadrature nodes and their corresponding weights. The network outputs x-coordinates, y-coordinates, and weights explicitly. This formulation is close to classical quadrature definitions and is easy to interpret. Training scripts in this folder handle dataset loading, training with a combined node and weight loss, checkpointing, and final model export. The testing script evaluates integration accuracy against Algoim reference data and produces error statistics and visualizations.



* The Scalling\_FNN folder implements an alternative formulation. Instead of predicting quadrature nodes directly, the network predicts transformation parameters. These parameters describe how a reference quadrature rule should be scaled and shifted in space. In the simplest version, only scaling factors in x and y are predicted in one version, in the updated version for a complete transformation it predicts scaling factors x and y and Center shifts x and y as well. Quadrature nodes are reconstructed later from these parameters. This approach reduces the dimensionality of the learning problem and generally leads to more stable training behavior.



All scripts are designed to run on CPU by default. Checkpointing is intentionally explicit to allow training to be resumed or analyzed later. The Classic\_FNN approach is explicit and interpretable but heavier, while the Scalling\_FNN approach is more compact and robust and is better suited for deployment scenarios. 





**Scaling-Based Quadrature Reconstruction – MATLAB Interface(scalling\_ML\_matlab)**



This folder contains MATLAB scripts that use a trained scaling-based neural network model to reconstruct quadrature nodes and weights and to validate the resulting quadrature rules. The main purpose of this folder is to provide a lightweight MATLAB interface for models that were trained in Python and exported (via ONNX). **The MATLAB code does not train neural networks**. It only performs inference, reconstruction, and verification.



The workflow is centered around predicting scaling and center parameters using a neural network and then reconstructing physical quadrature nodes and weights from a reference quadrature rule. This mirrors the Scalling\_FNN pipeline used during training in Python. The script scalling\_ML.m acts as the main entry point. It loads the trained scaling model, prepares the polynomial input data, runs inference to obtain scaling and center parameters, and passes these parameters to the reconstruction routines. The output of this script is a set of quadrature nodes and weights suitable for numerical integration.



The function reconstruct\_nodes\_and\_weights.m contains the core reconstruction logic. Given predicted scaling factors, center shifts, and a reference quadrature rule, it computes the final node positions and corresponding weights. This step translates the abstract network outputs into physically meaningful quadrature rules. The script test.m provides a simple validation setup. It applies the reconstructed quadrature rule to test integrals and compares the result against reference or analytical values. This script is intended for sanity checks and regression testing rather than large-scale evaluation. Training and model definition happen entirely in Python, while MATLAB is used here for reconstruction, testing, and integration into existing numerical workflows.

