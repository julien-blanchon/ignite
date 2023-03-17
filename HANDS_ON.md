# Personal (Julien Blanchon) Hands On with Ignite

> This document is a personal hands on with Pytorch Ignite. It is not a tutorial, it is not a guide, it is not a reference. It is just a personal log of what I did to guide my learning of Ignite.

See CONTRIBUTING.md for styling and ignite good practice !

## Philosophy

Structure:

- ignite - Core library files
- engine - Module containing core classes like Engine, Events, State.
- handlers - Module containing out-of-the-box handlers
- metrics - Module containing out-of-the-box metrics
- contrib - Contrib module with other metrics, handlers classes that may require additional dependencies
- distributed - Module with helpers for distributed computations

### The Engine object

The Engine object is the core of Ignite. It is a state machine that runs a given `process_function` over each batch of a dataset, emitting Events as it goes.

It can be considered like a training loop. It takes a `train_step` as an argument and runs it over each batch of the dataset, triggering events as it goes.

```raw
process_function (Callable[[Engine, Any], Any]) – A function receiving a handle to the engine and the current batch in each iteration, and returns data to be stored in the engine’s state.
```

```python
process_function: Callable[[Engine, BatchType], ReturnType]
```

> **Question**:
> Static/fixed BatchType and ReturnType or dynamic?
> I really like the idea of making static BatchType and ReturnType using Generics types.

The `.run(dataset)` methods takes a dataset and runs the `process_function` over each batch of the dataset.

Engine is also very usefull as this object is always passed to any king of handler function such that could be triggered at any time during the training loop using an event.

See this `log_training_loss` example:

```python
@trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
def log_training_loss(engine):
    print(f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.2f}")
```

The engine object can give context about the current state of the training `state.output` for example for the loss.

😕 I'm not a big fan of the dynamically typed approach of Ignite. I would prefer to have a static type for the ouput and the batch for example.

For example in my project I like to differentiate differents task and dataset using using a specific BatchType. For example for a satellite image segmentation task I would like to have a `BatchType` that contains the image and the mask. With something like a `NamedTuple`, `TypedDict` ... and is specific to the dataset.
And the same for the output of the model. I would like to have a `ReturnType` that contains the loss and the prediction and his specific to the task.
This help to have specific handler for each task and dataset, and to easily understand what is the input and the output.

### The Handlers object

Handlers - Functions that can be triggered when a certain Event is emitted by the Engine. Ignite has a long list of pre defined Handlers such as checkpoint, early stopping, logging and built-in metrics, see ignite-handlers.

#### Metrics object

Metrics seems to be attachable to Engine objects.

See `Accuracy().attach(engine, "val_acc")`

With it's not the opposite, Accuracy being attached to the engine ?

#### The ProgressBar object

With it's not the opposite, ProgressBar being attached to the engine ?

### The Events object

Events are emitted by the Engine when it reaches a specific point in the run/training.

Attachable using decorators at definition time:
`@trainer.on(Events.ITERATION_COMPLETED(every=100))`.

You could create a custom event using:
```python
class BackpropEvents(EventEnum):
    BACKWARD_STARTED = 'backward_started'
    BACKWARD_COMPLETED = 'backward_completed'
    OPTIM_STEP_COMPLETED = 'optim_step_completed'
```

And fire at any time them using

```python
engine.fire_event(BackpropEvents.OPTIM_STEP_COMPLETED)
```

Events sould be registered using `trainer.register_events(*BackpropEvents)`

### Distributed aspect

To handle distributed training, Ignite provide some helper function in `ignite.distributed`. In particular the `auto_dataloader`, `auto_model` and `auto_optim` that will handle the distributed aspect of each element.

See <https://pytorch-ignite.ai/blog/distributed-made-easy-with-ignite/>

## Good issues

See `label:"help wanted" ` and `label:"good first issue"`.

- [ ] ClearML support for configuration #2888 <https://github.com/pytorch/ignite/issues/2888>
- [ ] Top-K precision/recall multilabel metrics for ranking task #467 <https://github.com/pytorch/ignite/issues/467>
- [ ] 
