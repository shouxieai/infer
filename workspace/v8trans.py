import onnx
import onnx.helper as helper
import sys
import os

def main():

    if len(sys.argv) < 2:
        print("Usage:\n python v8trans.py yolov8n.onnx")
        return 1

    file = sys.argv[1]
    if not os.path.exists(file):
        print(f"Not exist path: {file}")
        return 1

    prefix, suffix = os.path.splitext(file)
    dst = prefix + ".transd" + suffix

    model = onnx.load(file)
    node  = model.graph.node[-1]

    old_output = node.output[0]
    node.output[0] = "pre_transpose"

    for specout in model.graph.output:
        if specout.name == old_output:
            shape0 = specout.type.tensor_type.shape.dim[0]
            shape1 = specout.type.tensor_type.shape.dim[1]
            shape2 = specout.type.tensor_type.shape.dim[2]
            new_out = helper.make_tensor_value_info(
                specout.name,
                specout.type.tensor_type.elem_type,
                [0, 0, 0]
            )
            new_out.type.tensor_type.shape.dim[0].CopyFrom(shape0)
            new_out.type.tensor_type.shape.dim[2].CopyFrom(shape1)
            new_out.type.tensor_type.shape.dim[1].CopyFrom(shape2)
            specout.CopyFrom(new_out)

    model.graph.node.append(
        helper.make_node("Transpose", ["pre_transpose"], [old_output], perm=[0, 2, 1])
    )

    print(f"Model save to {dst}")
    onnx.save(model, dst)
    return 0

if __name__ == "__main__":
    sys.exit(main())