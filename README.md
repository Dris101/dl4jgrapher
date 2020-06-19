# dl4jgrapher
Classes to generate Graphviz DOT files from DL4J MultiLayerNetworks and ComputationGraphs

For prototyping for example, with VS Code as your IDE, you can use João Pinto's excellent Graphviz (dot) language support extension (https://github.com/joaompinto/vscode-graphviz) to preview the generated DOT file alongside the dl4j code. You can also use Graphviz tools (https://graphviz.org/) to generate files in formats such as pdf, png, svg etc. from the DOT file for display / publication purposes.

```powershell
C:\Graphviz\bin\dot.exe -Tsvg alexnet.dot -o alexnet.svg
```

# Scala Examples

## MultiLayerNetwork

### Code

```scala
import org.deeplearning4j.zoo.model.AlexNet
import com.drissoft.dl4jgrapher._

val h = 224
val w = 224
val c = 3
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
```

### Output
![AlexNet](./alexnet.svg)

## ComputationGraph

### Code

```scala
import org.deeplearning4j.zoo.model.ResNet50
import com.drissoft.dl4jgrapher._

val h = 224
val w = 200
val c = 3
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
```

### Output
![ResNet50](./resnet.svg)