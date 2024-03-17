# Tweet Sentiment Analysis

## DockerFile

1. Build the docker image: `sudo docker build -t "base" .`.
2. Instansiate a container for training `sudo docker run -d --name <container_name> base`.

### Plan

https://chat.openai.com/c/f25dc443-5ed4-4686-9370-f3484582e4d3

1. Fix the pre-trained language model, maybe explore outside HuggingFace transformers, such as Llama.
2. Create a variety of adaptor architectures. Construct a docker image/container which takes in a parameter (should be able to be the same image) which takes a model architecture and does cross-validation. Outputs the best model parameters and trains model finally. Use CUDA in training and spin up the adapters in parallel.
3. Tokenisation and Vector DB storage, a way to access the db in parallel, cloud hosted database/DBAAS (database as a service).
4. An inference script (in its own container), which given an input returns the sentiment. Model quantisation of base model in inference. "dynamic, static (post-training), and quantization-aware training. For most scenarios, post-training quantization offers a good balance between implementation simplicity and performance gains."
    a. Validate the Quantized Model: It's important to test the quantized model to ensure that the loss in precision hasn't significantly affected its accuracy or the quality of its predictions. Run it against your validation set or a representative sample of your data.
    b. Integrate Quantization into the Inference Script: Adjust your inference script to load the quantized model and perform inference using it. Ensure that any necessary preprocessing or postprocessing steps are compatible with the quantized model's requirements.
    c. Deployment: When deploying your inference container, ensure that the environment (e.g., cloud instance, edge device) supports running the quantized model efficiently. Some environments offer hardware acceleration for quantized models, such as GPUs or TPUs, which can further enhance performance.
5. A script that listens to certain Twitter accounts and the inference container is instanisatied on a tweet, classifies and comments the sentiment.
