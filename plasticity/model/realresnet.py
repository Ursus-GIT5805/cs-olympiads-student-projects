from flax import nnx  # The Flax NNX API.
from functools import partial
import optax

import tensorflow_datasets as tfds  # TFDS to download MNIST.
import tensorflow as tf  # TensorFlow / `tf.data` operations.

tf.random.set_seed(0)  # Set the random seed for reproducibility.

train_steps = 1200
eval_every = 200
batch_size = 128

train_ds: tf.data.Dataset = tfds.load('mnist', split='train')
test_ds: tf.data.Dataset = tfds.load('mnist', split='test')

train_ds = train_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # normalize train set
test_ds = test_ds.map(
  lambda sample: {
    'image': tf.cast(sample['image'], tf.float32) / 255,
    'label': sample['label'],
  }
)  # Normalize the test set.

# Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
train_ds = train_ds.repeat().shuffle(1024)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
train_ds = train_ds.batch(batch_size, drop_remainder=True).take(train_steps).prefetch(1)
# Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
test_ds = test_ds.batch(batch_size, drop_remainder=True).prefetch(1)

class Residual(nnx.Module):
    """The residual block of ResNet"""

    def __init__(self, width, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=width,out_features=256,rngs=rngs)
        self.batch1 = nnx.BatchNorm(num_features=256,rngs=rngs)
        self.linear2 = nnx.Linear(in_features=256,out_features=width,rngs=rngs)

    def __call__(self, orx):
        x = nnx.relu(self.batch1(self.linear1(orx)))
        x = self.linear2(x)
        return nnx.relu(x+orx)


class Network(nnx.Module):
    def __init__(self, input_size, width, output_size, depth, *, rngs: nnx.Rngs):
        self.linear1 =  nnx.Linear(in_features=input_size, out_features=width, rngs=rngs)
        self.blocks = [Residual(width, rngs=rngs)
                    for _ in range(depth)]
        self.linear2 = nnx.Linear(in_features=width,out_features=output_size,rngs=rngs)
    def __call__(self, x):
        x = x.reshape(x.shape[0], -1)
        x = nnx.relu(self.linear1(x))
        for res in self.blocks:
            x = res(x)
        # Return logits (no softmax here)
        x = self.linear2(x)
        return x

model = Network(784, 256,10,6,rngs=nnx.Rngs(0))
# nnx.display(model)


learning_rate = 3e-3
momentum = 0.9

optimizer = nnx.Optimizer(
  model, optax.adamw(learning_rate, momentum, weight_decay=1e-4), wrt=nnx.Param
)
metrics = nnx.MultiMetric(
  accuracy=nnx.metrics.Accuracy(),
  loss=nnx.metrics.Average('loss'),
)

nnx.display(optimizer)

def loss_fn(model: Network, batch):
  logits = model(batch['image'])                # logits, not probabilities
  loss = optax.softmax_cross_entropy_with_integer_labels(
      logits=logits, labels=batch['label']
  ).mean()
  return loss, logits
@nnx.jit
def train_step(model: Network, optimizer: nnx.Optimizer, metrics: nnx.MultiMetric, batch):
  """Train for a single step."""
  grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
  (loss, logits), grads = grad_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.
  optimizer.update(model,grads)  # In-place updates.

@nnx.jit
def eval_step(model: Network, metrics: nnx.MultiMetric, batch):
  loss, logits = loss_fn(model, batch)
  metrics.update(loss=loss, logits=logits, labels=batch['label'])  # In-place updates.



metrics_history = {
  'train_loss': [],
  'train_accuracy': [],
  'test_loss': [],
  'test_accuracy': [],
}
num_epochs = 30
train_steps_per_epoch = 60000 // batch_size  # MNIST train set size

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    metrics.reset()

    # Build the dataset for this epoch (shuffle each epoch)
    epoch_train_ds = (
        tfds.load('mnist', split='train')
        .map(lambda sample: {
            'image': tf.cast(sample['image'], tf.float32) / 255,
            'label': sample['label']
        })
        .shuffle(1024)
        .batch(batch_size, drop_remainder=True)
    )

    for step, batch in enumerate(epoch_train_ds.as_numpy_iterator()):
        model.train()
        train_step(model, optimizer, metrics, batch)

    # End of epoch — log train metrics
    train_metrics = metrics.compute()
    print(f"Train loss: {train_metrics['loss']:.4f}, "
          f"acc: {train_metrics['accuracy']:.4f}")
    metrics.reset()

    # Evaluation
    model.eval()
    for batch in test_ds.as_numpy_iterator():
        eval_step(model, metrics, batch)
    test_metrics = metrics.compute()
    print(f"Test loss: {test_metrics['loss']:.4f}, "
          f"acc: {test_metrics['accuracy']:.4f}")
    metrics.reset()

