import tensorflow as tf



# # 运行图并更新变量
# with tf.compat.v1.Session() as sess:
#     sess.run(init_op)
#     for _ in range(5):
#         sess.run(update_op)
#         print("Updated value:", my_variable.eval())
    
#     # 保存模型为SavedModel格式
#     tf.saved_model.save(sess, "./saved_model/")
#     print("Model saved as SavedModel.")


g = tf.Graph()

with g.as_default():
    # 创建一个可训练的变量
    my_variable = tf.Variable(0.0, name="my_variable")

    # 在某个操作中使用这个变量
    update_op = my_variable.assign_add(1.0)

    # 初始化变量
    init_op = tf.compat.v1.global_variables_initializer()
    
gdef = g.as_graph_def()
outputName = 'demo'
with open("./%s.def" % outputName, 'w') as f:
    f.write(gdef.SerializeToString())
with open("./%s.pbtxt" % outputName, 'w') as f:
    f.write(str(gdef))