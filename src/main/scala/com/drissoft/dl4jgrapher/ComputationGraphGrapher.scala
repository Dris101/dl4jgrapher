package com.drissoft.dl4jgrapher

import scala.collection.JavaConverters._
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.conf.inputs.InputType

class ComputationGraphGrapher(net: ComputationGraph) {
  def getGrapher(inputs: Seq[INDArray], inputTypes: Seq[InputType]) = {
    val activations = net
      .feedForward(inputs.toArray, false)
      .asScala
      .toMap

    new DotGrapher(net.getConfiguration(), activations, inputTypes)
  }
}
