# dl4jgrapher

A listener which creates Graphviz DOT files of a DL4J ComputationGraph

For example if you use VS Code as your IDE, you can use João Pinto's excellent Graphviz (dot) language support extension (https://github.com/joaompinto/vscode-graphviz) to preview the generated dot file alongside the dl4j code.

# Scala Example

```scala
import com.drissoft.dl4jgrapher.listeners._

val h = 224
val w = 200
val c = 3
val inputTypes = List(new InputType.InputTypeConvolutional(h, w, c))

// Build ResNet50
val net = ResNet50
  .builder()
  .numClasses(10)
  .seed(1234L)
  .build()
  .init()

// Attach listener
val diag = new DotGraphListener(net, inputTypes)
net.setListeners(diag)

// Get output and fit using it (only need the output shape to be consistent)
val f      = Nd4j.rand(1, c, h, w)
val result = net.output(f);
net.fit(Array(f), result)

// Output the DOT file
diag.toDotFile(java.nio.file.Paths.get("resnet.dot"))
```

# Example Output
![Image of network graph](resnet.svg)
