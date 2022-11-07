
"""Experiment to train and evaluate the TabNet model on Intellizenz veranstaltung segment"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import app
import data_helper_intellizenz
import numpy as np
import tabnet_model
import tensorflow as tf
import pandas as pd


# Run Tensorflow on GPU 0
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training parameters
TRAIN_FILE = "data/train_intellizenz.csv"
VAL_FILE = "data/val_intellizenz.csv"
TEST_FILE = "data/test_intellizenz.csv"
MAX_STEPS = 1000000
DISPLAY_STEP = 5000
VAL_STEP = 10000
SAVE_STEP = 40000
INIT_LEARNING_RATE = 0.02
DECAY_EVERY = 500
DECAY_RATE = 0.95
BATCH_SIZE = 16384
SPARSITY_LOSS_WEIGHT = 0.0001
GRADIENT_THRESH = 2000.0
SEED = 1


def main(unused_argv):
    # Fix random seeds
    # tf.set_random_seed(SEED)
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    # Define the TabNet model
    tabnet_intellizenz = tabnet_model.TabNet(
        columns=data_helper_intellizenz.get_columns(),
        num_features=data_helper_intellizenz.NUM_FEATURES,
        feature_dim=128,
        output_dim=64,
        num_decision_steps=6,
        relaxation_factor=1.5,
        batch_momentum=0.7,
        virtual_batch_size=512,
        num_classes=data_helper_intellizenz.NUM_CLASSES)

    column_names = sorted(data_helper_intellizenz.FEATURE_COLUMNS)
    print("Ordered column names, corresponding to the indexing in Tensorboard visualization")

    for fi in range(len(column_names)):
        print(str(fi) + " : " + column_names[fi])

    # Input sampling
    train_batch = data_helper_intellizenz.input_fn(
        TRAIN_FILE, num_epochs=100000, shuffle=True, batch_size=BATCH_SIZE)
    val_batch = data_helper_intellizenz.input_fn(
        VAL_FILE,
        num_epochs=10000,
        shuffle=False,
        batch_size=data_helper_intellizenz.N_VAL_SAMPLES)
    test_batch = data_helper_intellizenz.input_fn(
        TEST_FILE,
        num_epochs=10000,
        shuffle=False,
        batch_size=data_helper_intellizenz.N_TEST_SAMPLES)

    # train_iter = train_batch.make_initializable_iterator()
    # val_iter = val_batch.make_initializable_iterator()
    # test_iter = test_batch.make_initializable_iterator()

    train_iter = iter(train_batch)
    val_iter = iter(val_batch)
    test_iter = iter(test_batch)

    feature_train_batch, label_train_batch = train_iter.get_next()
    feature_val_batch, label_val_batch = val_iter.get_next()
    feature_test_batch, label_test_batch = test_iter.get_next()

    print(label_train_batch)

    # Define the model and losses

    encoded_train_batch, total_entropy = tabnet_intellizenz.encoder(
        feature_train_batch, reuse=False, is_training=True)

    #   logits - interpreting inputs as log probabilities  
    logits_orig_batch, _ = tabnet_intellizenz.classify(
        encoded_train_batch, reuse=False)

    #   softmax_cross_entropy_with_logits - applying cross entropy loss after applying softmax functions to the logits
    #   sparse_softmax_cross_entropy_with_logits - if each class has only one label and the class labels are not required to be converted to one-hot array
    #   For sparse_softmax_cross_entropy_with_logits, labels must have the shape [batch_size] and 
    #   the dtype int32 or int64. Each label is an int in range [0, num_classes-1].
    softmax_orig_key_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits_orig_batch, labels=label_train_batch))

    train_loss_op = softmax_orig_key_op + SPARSITY_LOSS_WEIGHT * total_entropy
    tf.summary.scalar("Total loss", train_loss_op)

    # Optimization step
    # global_step = tf.train.get_or_create_global_step()
    # learning_rate = tf.train.exponential_decay(
    #     INIT_LEARNING_RATE,
    #     global_step=global_step,
    #     decay_steps=DECAY_EVERY,
    #     decay_rate=DECAY_RATE)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    
    # with tf.control_dependencies(update_ops):
    #     gvs = optimizer.compute_gradients(train_loss_op)
    #     capped_gvs = [(tf.clip_by_value(grad, -GRADIENT_THRESH,
    #                                     GRADIENT_THRESH), var) for grad, var in gvs]
    #     train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)

    global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate = tf.compat.v1.train.exponential_decay(
        INIT_LEARNING_RATE,
        global_step=global_step,
        decay_steps=DECAY_EVERY,
        decay_rate=DECAY_RATE)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)

    print('Total entropy: ',total_entropy)
    print('Aggregate output: ',encoded_train_batch)

    tf.compat.v1.disable_v2_behavior()

    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(train_loss_op)
        print(train_op)
        # gvs = optimizer.compute_gradients(train_loss_op,[encoded_train_batch, total_entropy])
        # print('The grads: ',gvs)
        # capped_gvs = [(tf.clip_by_value(grad, -GRADIENT_THRESH,
        #                                 GRADIENT_THRESH), var) for grad, var in gvs]
        # train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)


    # Optimization step
    # global_step = tf.compat.v1.train.get_or_create_global_step()	
    # learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=INIT_LEARNING_RATE,
    #                                                                  decay_steps=DECAY_EVERY,decay_rate=DECAY_RATE)(global_step)
    # # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    # gvs = optimizer.compute_gradients(train_loss_op)
    
    # # Process the gradients, for example cap them, etc.
    # # capped_grads = [MyCapper(g) for g in grads]
    # capped_gradss = [(tf.clip_by_value(grad, -GRADIENT_THRESH,
    #                                 GRADIENT_THRESH), var) for grad, var in gvs]
    # # processed_grads = [process_gradient(g) for g in grads]

    # # Ask the optimizer to apply the processed gradients.
    # # train_op = optimizer.apply_gradients(zip(capped_gradss, encoded_train_batch))
    # train_op = optimizer.apply_gradients(capped_gradss, global_step=global_step)

    # optimizer.minimize(train_loss_op, var_list=encoded_train_batch)
    # train_op = optimizer.minimize(train_loss_op, var_list=encoded_train_batch)
    # train_op.run()

    # # Compute the gradients for a list of variables.
    # with tf.GradientTape() as tape:
    #     grads = tape.gradient(train_loss_op, trainable_vars)
    #     print('The loss : ', train_loss_op)
    #     print('The softmax loss : ', softmax_orig_key_op)
    #     # Compute the gradients for a list of variables.
    #     # grads = optimizer.minimize(train_loss_op, encoded_train_batch,tape=tape)
    #     print('The gradients : ', grads)


    #     # Process the gradients, for example cap them, etc.
    #     # capped_grads = [MyCapper(g) for g in grads]
    #     capped_grads = [(tf.clip_by_value(grad, -GRADIENT_THRESH,
    #                                     GRADIENT_THRESH), var) for grad, var in grads]
    #     # processed_grads = [process_gradient(g) for g in grads]

    #     # Ask the optimizer to apply the processed gradients.
    #     train_op = optimizer.apply_gradients(zip(capped_grads, encoded_train_batch))



    # Model evaluation

    # Validation performance
    encoded_val_batch, _ = tabnet_intellizenz.encoder(
        feature_val_batch, reuse=True, is_training=False)

    _, prediction_val = tabnet_intellizenz.classify(
        encoded_val_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_val, 1), dtype=tf.int32)
    val_eq_op = tf.equal(predicted_labels, label_val_batch)
    val_acc_op = tf.reduce_mean(tf.cast(val_eq_op, dtype=tf.float32))
    tf.summary.scalar("Val accuracy", val_acc_op)

    # Test performance
    encoded_test_batch, _ = tabnet_intellizenz.encoder(
        feature_test_batch, reuse=True, is_training=False)

    _, prediction_test = tabnet_intellizenz.classify(
        encoded_test_batch, reuse=True)

    predicted_labels = tf.cast(tf.argmax(prediction_test, 1), dtype=tf.int32)
    test_eq_op = tf.equal(predicted_labels, label_test_batch)
    test_acc_op = tf.reduce_mean(tf.cast(test_eq_op, dtype=tf.float32))
    tf.summary.scalar("Test accuracy", test_acc_op)

    # Training setup
    model_name = "tabnet_intellizenz_model"
    init = tf.compat.v1.initialize_all_variables()
    init_local = tf.compat.v1.local_variables_initializer()
    init_table = tf.compat.v1.tables_initializer(name="Initialize_all_tables")
    saver = tf.compat.v1.train.Saver()
    summaries = tf.compat.v1.summary.merge_all()

    with tf.compat.v1.Session() as sess:
        summary_writer = tf.compat.v1.summary.FileWriter("./tflog/" + model_name, sess.graph)

        sess.run(init)
        sess.run(init_local)
        sess.run(init_table)
        sess.run(train_iter.initializer)
        sess.run(val_iter.initializer)
        sess.run(test_iter.initializer)

        for step in range(1, MAX_STEPS + 1):
            if step % DISPLAY_STEP == 0:
                _, train_loss, merged_summary = sess.run(
                    [train_op, train_loss_op, summaries])
                summary_writer.add_summary(merged_summary, step)
                print("Step " + str(step) + " , Training Loss = " +
                    "{:.4f}".format(train_loss))
            else:
                _ = sess.run(train_op)

            if step % VAL_STEP == 0:
                feed_arr = [
                    vars()["summaries"],
                    vars()["val_acc_op"],
                    vars()["test_acc_op"]
                ]

                val_arr = sess.run(feed_arr)
                merged_summary = val_arr[0]
                val_acc = val_arr[1]

                print("Step " + str(step) + " , Val Accuracy = " +
                    "{:.4f}".format(val_acc))
                summary_writer.add_summary(merged_summary, step)

            if step % SAVE_STEP == 0:
                saver.save(sess, "./checkpoints/" + model_name + ".ckpt")


if __name__ == "__main__":
    app.run(main)
