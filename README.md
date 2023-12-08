# HIG: Hierarchical Interlacement Graph Approach to Scene Graph Generation in Video Understanding


[**HIG: Hierarchical Interlacement Graph Approach to Scene Graph Generation in Video Understanding**](./static/pdfs/main_paper.pdf)

[Trong-Thuan Nguyen](https://scholar.google.com/citations?user=ty0Njf0AAAAJ&hl=vi&authuser=1), [Pha Nguyen](https://pha-nguyen.github.io/), [Khoa Luu](https://scholar.google.com/citations?user=JPAl8-gAAAAJ)


Abstract
--------

Visual interactivity understanding within visual scenes presents a significant challenge in computer vision. 
Existing methods focus on complex interactivities while leveraging a simple relationship model. 
These methods, however, struggle with a diversity of appearance, situation, position, interaction, and relation in videos. 
This limitation hinders the ability to fully comprehend the interplay within the complex visual dynamics of subjects. 
In this paper, we delve into interactivities understanding within visual content by deriving scene graph representations from dense interactivities among humans and objects. 
To achieve this goal, we first present a new dataset containing <i>Appearance-Situation-Position-Interaction-Relation</i> predicates, named <i>ASPIRe</i>, 
offering an extensive collection of videos marked by a wide range of interactivities. Then, we propose a new approach named <i>Hierarchical Interlacement Graph (HIG)</i>, 
which leverages a unified layer and graph within a hierarchical structure to provide deep insights into scene changes across five distinct tasks. 
Our approach demonstrates superior performance to other methods through extensive experiments conducted in various scenarios.

Introduction
------------

We introduce the new <i>ASPIRe</i> dataset to Visual Interactivity Understanding.
The diversity of the <i>ASPIRe</i> dataset is showcased through its wide range of scenes and settings, distributed in seven scenarios.

Examples of annotations found on the ASPIRe dataset.



Annotations
-----------

v1.0:
-----

*   The category **name**, **bbox**,  **segmentation** and **track_id** compatible with that [TAO](https://taodataset.org/) dataset.
    *   Training set: [\[Train annotations\]](./annotations/v1.0/train.json), 
    *   Testing set: [\[Test annotations\]](./annotations/v1.0/test.json),


Licensing:
----------

The annotations of <i>ASPIRe</i> and the original source videos are released under a <a
href="https://creativecommons.org/licenses/by-nc-sa/3.0/" target="_blank">CC BY-NC-SA
3.0</a> license per their creators. See <a href="https://motchallenge.net/"
target="_blank">motchallenge.net</a> for details.



This page was built using the [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).  
This website is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-sa/4.0/).
