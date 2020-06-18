package com.drissoft.dl4jgrapher.listeners

import org.scalatest._
import org.scalatest.wordspec._
import org.scalatest.matchers.must.Matchers
import org.deeplearning4j.zoo.model.{ResNet50, AlexNet}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.nd4j.linalg.factory.Nd4j

class DotGraphListenerSpec extends AnyWordSpec with Matchers {
  "A ResNet" must {
    "generate a graph" in {
      val h          = 224
      val w          = 200
      val c          = 3
      val inputTypes = List(new InputType.InputTypeConvolutional(h, w, c))

      // Build ResNet50
      val net = ResNet50
        .builder()
        .numClasses(10)
        .build()
        .init()

      // Attach listener
      val diag = new DotGraphListener(net, inputTypes)
      net.setListeners(diag)

      // Get output and fit using it (only need the output shape to be consistent)
      // so that onForwardPass is called on the listener
      val f      = Nd4j.rand(1, c, h, w)
      val result = net.output(f);
      net.fit(Array(f), result)

      // Output the DOT file
      diag.toDotFile(java.nio.file.Paths.get("resnet.dot"))
    }
  }

  "AlexNet" must {
    "generate a graph" in {
      val h          = 224
      val w          = 224
      val c          = 3
      val inputTypes = List(new InputType.InputTypeConvolutional(h, w, c))

      // Build ResNet50
      val net = AlexNet
        .builder()
        .numClasses(10)
        .build()
        .init()

      // Attach listener. Have to cast MultiLayerNetwork to ComputationGraph
      val diag = new DotGraphListener(net.toComputationGraph(), inputTypes)
      net.setListeners(diag)

      // Get output and fit using it (only need the output shape to be consistent)
      // so that onForwardPass is called on the listener
      val f      = Nd4j.rand(1, c, h, w)
      val result = net.output(f)
      net.fit(f, result)

      // Output the DOT file
      diag.toDotFile(java.nio.file.Paths.get("alexnet.dot"))
    }
  }
}
