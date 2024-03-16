import os
import pandas as pd

import jax
import jax.numpy as jnp
import flax.linen as nn 
from jax.nn.initializers import lecun_normal
from jax import value_and_grad
import optax

from transformers import BertTokenizerFast, FlaxBertModel

train_df = pd.read_csv("./twitter_training.csv", names=["tweet_id", "entity", "sentiment", "content"])
train_df = train_df.dropna()
train_df['class'], classes = pd.factorize(train_df['sentiment'])
num_classes = len(classes)

embedding_model = FlaxBertModel.from_pretrained('bert-base-cased')
tokeniser = BertTokenizerFast.from_pretrained('bert-base-cased')

def sample(df : pd.DataFrame, batch_size : int = 128):
    frequencies = 1.0 / df['class'].value_counts()
    weights = df['class'].map(frequencies)
    sample = df.sample(batch_size, replace = True, weights = frequencies)
    return process_sample(sample)

def process_sample(sample_df : pd.DataFrame):
    inputs = tokeniser.batch_encode_plus(sample_df['content'].tolist(), add_special_tokens=True, truncation=True, padding=True, return_tensors='jax')
    outputs = jax.nn.one_hot(sample_df['class'].values, num_classes=num_classes)
    return embedding_model(**inputs).last_hidden_state, outputs

class Model(nn.Module):

    @nn.compact
    def __call__(self, embedding):
        x = nn.Dense(features = 128, kernel_init = lecun_normal())(embedding)
        x = nn.relu(x)
        x = jnp.mean(x, axis=-1)
        x = nn.Dense(features = 128, kernel_init = lecun_normal())(x)
        x = nn.relu(x)
        x = nn.Dense(features = 4, kernel_init = lecun_normal())(x)
        return nn.softmax(x)


model = Model()
rng, init_rng = jax.random.split(jax.random.PRNGKey(42), 2)
example_input, example_output = sample(train_df, batch_size = 128)

params = model.init(init_rng, example_input)
model.apply(params, example_input).shape

optimiser = optax.chain(optax.clip(1.0), optax.adam(learning_rate=1e-4))
optimiser_state = optimiser.init(params)

def cross_entropy_loss(params, batch_inputs, batch_outputs):
    model_outputs = model.apply(params, batch_inputs)
    return -jnp.mean(jnp.sum(batch_outputs * jnp.log(model_outputs), axis = 1))

num_epochs = 1
batch_size = 128

for epoch in range(num_epochs):
    batch_losses = []
    for batch in range(1): # train_df.shape[0] // batch_size):
        batch_inputs, batch_outputs = sample(train_df, batch_size=128)
        loss, grads = jax.value_and_grad(cross_entropy_loss)(params, batch_inputs, batch_outputs)
        updates, optimiser_state = optimiser.update(grads, optimiser_state)
        params = optax.apply_updates(params, updates)
        batch_losses.append(loss)
    print(jnp.mean(jnp.array(batch_losses)))
