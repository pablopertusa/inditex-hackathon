

In a two-tower architecture, each tower is a neural network that processes either query or candidate input features to produce an embedding representation of those features. Because the embedding representations are simply vectors of the same length, we can compute the dot product between these two vectors to determine how close they are. This means the orientation of the embedding space is determined by the dot product of each <query, candidate> pair in the training examples. En este caso, query es el usuario y candidate es un producto (partnumber). https://cloud.google.com/blog/products/ai-machine-learning/scaling-deep-retrieval-tensorflow-two-towers-architecture
https://blog.reachsumit.com/posts/2023/03/two-tower-model/
https://arxiv.org/pdf/1907.06902

Post de reddit con algunas sugerencias
https://www.reddit.com/r/recommendersystems/comments/1c89v8r/how_to_implement_two_tower_system/
Respuesta a este post
"
I would suggest the Tensorflow Recommenders library or the Two Tower implementation of the LibRecommender library. Both show examples on how to setup the model. Tensorflow is more flexible in how to setup the Query Model but can be a pain as well.
"

Blog que cuenta la implementacion
https://biarnes-adrien.medium.com/building-a-multi-stage-recommendation-system-part-1-2-ce006f0825d1





Predicciones:
    - Si está logeado tenemos historial de compras e interacciones (user_id)
    - Si no lo está tendremos que hacer la predicción en función de las últimas interacciones (session_id)
    - Si no está logeado ni tiene interacciones hay que hacer una predicción a ciegas
    ?- Cómo se pueden hacer predicciones en función de la sesión y del user_id