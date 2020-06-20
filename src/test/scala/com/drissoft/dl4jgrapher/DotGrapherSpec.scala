package com.drissoft.dl4jgrapher

import org.scalatest._
import org.scalatest.wordspec._
import org.scalatest.matchers.must.Matchers
import org.deeplearning4j.zoo.model.{ResNet50, AlexNet, InceptionResNetV1}
import org.deeplearning4j.nn.conf.inputs.InputType
import org.nd4j.linalg.factory.Nd4j

class DotGrapherSpec extends AnyWordSpec with Matchers {
  "ResNet50" must {
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

      val input   = Nd4j.rand(1, c, h, w)
      val grapher = new ComputationGraphGrapher(net).getGrapher(Array(input), inputTypes)

      // Output the DOT file
      grapher.writeDotFile(java.nio.file.Paths.get("resnet.dot"))
    }
  }

  "AlexNet" must {
    "generate a graph" in {
      val h         = 224
      val w         = 224
      val c         = 3
      val inputType = new InputType.InputTypeConvolutional(h, w, c)

      // Build AlexNet
      val net = AlexNet
        .builder()
        .numClasses(10)
        .build()
        .init()

      val input   = Nd4j.rand(1, c, h, w)
      val grapher = new MultiLayerNetworkGrapher(net).getGrapher(input, inputType)

      // Output the DOT file
      grapher.writeDotFile(java.nio.file.Paths.get("alexnet.dot"))
    }
  }

  "InceptionResNetV1" must {
    "generate a graph" in {
      val h          = 160
      val w          = 160
      val c          = 3
      val inputTypes = List(new InputType.InputTypeConvolutional(h, w, c))

      // Build InceptionResNetV1
      val net = InceptionResNetV1
        .builder()
        .numClasses(10)
        .build()
        .init()

      val input   = Nd4j.rand(1, c, h, w)
      val grapher = new ComputationGraphGrapher(net).getGrapher(Array(input), inputTypes)

      // Output the DOT file
      grapher.writeDotFile(java.nio.file.Paths.get("inception.dot"))
    }
  }
}
