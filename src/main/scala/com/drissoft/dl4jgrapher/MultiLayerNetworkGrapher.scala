package com.drissoft.dl4jgrapher

import scala.collection.JavaConverters._
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.inputs.InputType

class MultiLayerNetworkGrapher(net: MultiLayerNetwork) {
  def getGrapher(input: INDArray, inputType: InputType) = {
    val activations = net
      .feedForward(input)
      .asScala
      .zipWithIndex
      .map {
        case (a, 0) => ("in", a)
        case (a, i) => ((i - 1).toString, a)
      }
      .toMap

    new DotGrapher(net.toComputationGraph().getConfiguration(), activations, inputType :: Nil)
  }
}
