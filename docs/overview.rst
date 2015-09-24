Overview
========

.. uml::

   @startuml

   package "TAOConvert" #DDDDDD {

     abstract class Converter

     class Exporter

     class Mapping

     abstract class Module

     class LightCone

     class SED

     class Dust

     abstract class Validator

     abstract class Generator

     Converter o-- Module
     Converter .. Mapping
     Converter .. Exporter
     Module o-- Validator
     Module o-- Generator
     Module o-- Mapping
     Module <|-- LightCone
     Module <|-- SED
     Module <|-- Dust
   }

   Converter <|-- SAGEConverter

   @enduml

.. uml::

   @startuml
   skinparam groupInheritance 3

   package "TAOConvert" #DDDDDD {

     abstract class Validator

     class Required

     class TreeLocalIndex

     class Positive

     class NonZero

     class WithinRange

     class WithinCRange

     class Choice

     Validator <|-- Required
     Validator <|-- TreeLocalIndex
     Validator <|-- Positive
     Validator <|-- NonZero
     Validator <|-- WithinRange
     Validator <|-- WithinCRange
     Validator <|-- Choice
   }

   @enduml

.. uml::

   @startuml
   skinparam groupInheritance 3

   package "TAOConvert" #DDDDDD {

     abstract class Generator

     class GlobalIndices

     class TreeIndices

     class TreeLocalIndices

     class GlobalDescendants

     class DepthFirstOrdering

     Generator <|-- GlobalIndices
     Generator <|-- TreeIndices
     Generator <|-- TreeLocalIndices
     Generator <|-- GlobalDescendants
     Generator <|-- DepthFirstOrdering
   }

   @enduml

 ..
    Not sure why we need this, but keep one space before the ..
