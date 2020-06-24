package com.drissoft.dl4jgrapher

import java.nio.file._
import scala.util._
import scala.collection.JavaConverters._
import io.circe._
import io.circe.parser._
import io.circe.optics.JsonPath._
import org.deeplearning4j.nn.conf.inputs.InputType
import org.deeplearning4j.nn.conf.inputs.InputType._
import org.deeplearning4j.nn.conf.preprocessor._
import org.deeplearning4j.nn.conf.graph._
import org.nd4j.linalg.api.ndarray.INDArray
import org.deeplearning4j.nn.conf.InputPreProcessor
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration

class DotGrapher(conf: ComputationGraphConfiguration, activations: Map[String, INDArray], inputTypes: Seq[InputType]) {

  private case class VertexNode(
      id: String,
      label: String,
      nodeType: Option[String],
      activationType: Option[String],
      activationShape: List[Long],
      lossFunction: Option[String],
      preProcessor: Option[InputPreProcessor],
      nodeShape: String,
      color: String,
      backgroundColor: String
  )

  private val inputs       = conf.getNetworkInputs().asScala.toSet
  private val outputs      = conf.getNetworkOutputs().asScala.toSet
  private val vertexInputs = conf.getVertexInputs().asScala map { case (k, v) => k -> v.asScala }
  private val verticies    = conf.getVertices().asScala
  private val actTypes     = conf.getLayerActivationTypes(inputTypes: _*).asScala

  override def toString() = {
    s"""Inputs: ${inputs.mkString(",")}
         |Outputs: ${outputs.mkString(",")}
         |
         |Activation shapes:-
         |${activations.map { case (k, v) => (k -> v.shape().mkString(",")) }.mkString("\n")}""".stripMargin
  }

  private def vertexOutputs = {
    vertexInputs.toList flatMap {
      case (vertex, inputs) =>
        inputs map (_ -> vertex)
    }
  }.groupBy(_._1).map {
    case (input, outputs) =>
      (input -> outputs.map { _._2 })
  }

  def toDot() = {
    val sb = new StringBuilder("digraph ComputationGraph {\n")

    // Optics
    val _layerClass        = root.layer.`@class`.string
    val _activationClass   = root.layer.activationFn.`@class`.string
    val _lossFunctionClass = root.layer.lossFn.`@class`.string

    def makeNode(node: String) = {
      val nb       = new StringBuilder()
      val isOutput = outputs.contains(node)

      val bgcolor = if (inputs.contains(node)) {
        "red"
      } else if (isOutput) {
        "Yellow"
      } else {
        "darkslategrey"
      }

      val description: VertexNode = verticies
        .get(node)
        .map { v =>
          // Prettier layer names for layers with numbers
          val label = Try(node.toInt) match {
            case Success(l) => s"Layer $l"
            case _          => node
          }

          v match {
            case l: LayerVertex =>
              val json: Json = parse(l.getLayerConf().toJson()).getOrElse(Json.Null)
              val preProc    = l.getPreProcessor()
              VertexNode(
                node,
                label,
                _layerClass.getOption(json).map { _.split('.').last },
                _activationClass.getOption(json).map { _.split('.').last },
                activations(node).shape().toList,
                _lossFunctionClass.getOption(json).map { _.split('.').last },
                Option(preProc),
                "none",
                if (isOutput) "black" else "cornsilk",
                bgcolor
              )

            case g: GraphVertex =>
              VertexNode(
                node,
                label,
                Some(g.getClass().getName().split('.').last),
                None,
                activations(node).shape().toList,
                None,
                None,
                "none",
                "cornsilk",
                bgcolor
              )

            case _ =>
              VertexNode(
                node,
                label,
                None,
                None,
                activations(node).shape().toList,
                None,
                None,
                "none",
                "cornsilk",
                bgcolor
              )
          }
        }
        .getOrElse({
          // If there is no activation in the map, show empty placeholder
          val shape = activations.get(node).map { _.shape().toList }.getOrElse(List())
          VertexNode(node, node, None, None, shape, None, None, "none", "cornsilk", bgcolor)
        })

      nb.append(
        raw""""${description.id}" [shape=${description.nodeShape}, margin=0, label=<<table border="0" cellspacing = "0" cellborder="1">"""
      )

      description.preProcessor foreach { p =>
        val pText = p match {
          case rnnCnn: RnnToCnnPreProcessor =>
            val c = rnnCnn.getNumChannels()
            s"RNN to CNN (${rnnCnn.getInputHeight()} x ${rnnCnn.getInputWidth()}, $c chan${if (c > 1) "s"
            else ""}, ${rnnCnn.getRnnDataFormat()})"

          case ffRnn: FeedForwardToRnnPreProcessor => s"FF to RNN (${ffRnn.getRnnDataFormat()})"
          case rnnFf: RnnToFeedForwardPreProcessor => s"RNN to FF (${rnnFf.getRnnDataFormat()})"

          case cnnRnn: CnnToRnnPreProcessor =>
            val c = cnnRnn.getNumChannels()
            s"CNN to RNN (${cnnRnn.getInputHeight()} x ${cnnRnn.getInputWidth()}, $c chan${if (c > 1) "s"
            else ""}, ${cnnRnn.getRnnDataFormat()})"

          case cnnFf: CnnToFeedForwardPreProcessor =>
            val c = cnnFf.getNumChannels()
            s"CNN to FF (${cnnFf.getInputHeight()} x ${cnnFf.getInputWidth()}, $c chan${if (c > 1) "s" else ""}, ${cnnFf.getFormat})"
        }
        nb.append(s"""<tr><td bgcolor="darkolivegreen1">$pText</td></tr>\n""")
      }
      nb.append(
        raw"""<tr><td bgcolor="${description.backgroundColor}"><font color="${description.color}">${description.label}</font></td></tr>"""
      )

      description.nodeType foreach { t => nb.append(s"<tr><td>$t</td></tr>\n") }
      description.activationType foreach { t => nb.append(s"<tr><td>$t</td></tr>\n") }
      nb.append(s"<tr><td>[${description.activationShape.mkString(",")}]</td></tr>\n</table>>];\n")

      // Outgoing edges
      val edgeLabel = actTypes(node) match {
        case _: InputTypeRecurrent       => "RNN"
        case _: InputTypeFeedForward     => "FF"
        case _: InputTypeConvolutional   => "CNN"
        case _: InputTypeConvolutional3D => "CNN3D"
      }
      for {
        outs <- vertexOutputs.get(node)
        o    <- outs
      } nb.append(s""""$node" -> "$o" [label="  $edgeLabel"];\n""")

      description.lossFunction foreach { lf =>
        val lossNode = s"${node}_Loss"
        nb.append(s""""$node" -> "$lossNode";""")
        nb.append(s""""$lossNode" [label=$lf, style=filled, color=lightskyblue];""")
      }
      nb.append("\n")
      nb.result()
    }

    // Nodes
    for (node <- vertexOutputs.keys ++ outputs) {
      sb.append(makeNode(node))
    }

    sb.append("}")
    sb.result()
  }

  def writeDotFile(path: Path) = {
    Files.writeString(path, toDot())
  }
}
