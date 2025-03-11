// import logo from './logo.svg';
import './App.css';
import StandardGraph from './StandardGraph';
import logo from './Associated Logo.jpg';

function App() {
  return (
    <div className="App">
      <div className="App-header">
        <header className="Title">
          <img src={logo} />
          <p>New Madrid Forecast</p>
        </header>
        <StandardGraph width={600} height={400} origin='/noaa/gauge' title={'NOAA Predictions (Ft)'} axis={"Gauge Height (ft)"}></StandardGraph>
      </div>
    </div>
  );
}

export default App;
