from pathlib import Path
import tensorflow as tf
import keras.optimizers as optim
import keras.optimizers.schedules as scheduler
import keras.losses as losses

def train(
        model,
        train_config_dict: dict,
        input_dir: Path,
        output_dir: Path) -> tuple:
    '''
    Function that implements model training.

    :param model: tensorflow image classification model
    :param train_config_dict: configuration training dictionary
    :param train_loader: torch train dataloader
    :param valid_loader: torch valid dataloader
    :param output_dir: Path where trained model will be stored
    :return: Best and last metrics.
    '''

    train_ds = tf.keras.utils.image_dataset_from_directory(
        str(input_dir),
        validation_split=1-train_config_dict['train_split'],
        subset="training",
        seed=123,
        batch_size=train_config_dict['batch_size'])

    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(input_dir),
        validation_split=1-train_config_dict['train_split'],
        subset="validation",
        seed=123,
        batch_size=train_config_dict['batch_size'])

    class_names = train_ds.class_names

    #AUTOTUNE = tf.data.AUTOTUNE

    #train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    #val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    optimizer_name = train_config_dict['optimizer']['name']
    optimizer_params = train_config_dict['optimizer']['params']
    optimizer_params['params'] = model.parameters()
    train_config_dict['optimizer'] = getattr(
        optim, optimizer_name)(
        **optimizer_params)

    scheduler_name = train_config_dict['scheduler']['name']
    scheduler_params = train_config_dict['scheduler']['params']
    scheduler_params['optimizer'] = train_config_dict['optimizer']
    train_config_dict['scheduler'] = getattr(
        scheduler, scheduler_name)(
        **scheduler_params)

    train_config_dict['loss_function'] = getattr(losses, train_config_dict['loss_function'])

    model.compile(optimizer=train_config_dict['optimizer'],
                  lr_scheduler=train_config_dict['scheduler'],
                  loss=train_config_dict['loss_function'],
                  metrics=['accuracy'])

    model.summary()


    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=train_config_dict['epochs']
    )

