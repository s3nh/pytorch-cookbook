import React, { Component } from 'react'
import axios from 'axios'

class App extends Component {
  constructor() {
    super()

    this.state = {
      predictions: []
    }
  }

  fileChangedHandler = (e) => {
    const image = e.target.files[0]; const name = e.target.files[0].name

    const formData = new FormData()
    formData.append('image', image, name)

    axios.post('http://localhost:5000/api/v1/classify', formData)
    .then(resp => {
      this.setState(state => ({
        predictions: [...state.predictions, {
          key:  name,
          image: window.URL.createObjectURL(image),
          class: resp.data.prediction
        }]
      }))
    });

  }

  render() {
    return (
      <div className="container">
        <div className="row">
        {this.state.predictions.map(prediction => (
          <div className="col-4">
            <img className="img-responsive img-thumbnail" src={prediction.image}/>
            <p>{prediction.class}</p>
          </div>
        ))}
        </div>
        <hr/>
        <input type="file" onChange={this.fileChangedHandler}/>
      </div>
    )
  }
}

export default App
