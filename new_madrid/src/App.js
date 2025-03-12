// import logo from './logo.svg';
import './App.css';
import StandardGraph from './StandardGraph';
import PredictionGraph from './PredictionGraph';
import logo from './Associated Logo.jpg';

var origins = ['/observed/07024175/00065', '/noaa/gauge'];
var names = ['Observed Data', 'NOAA Predictions'];  
var colors = ['#8884d8', '#3355ff'];

function App() {
  return (
    <div className="App">
      <div className="App-header">
        <header className="Title">
          <img src={logo} />
          <p>New Madrid Forecast Ensemble</p>
        </header>
        <div className="GraphContainer">
          <PredictionGraph width={800} height={543} origins={origins} title={'New Madrid Gauge Forecast'} names={names} colors={colors} axis={"Gauge Height (ft)"}/>
          <div>
            <StandardGraph width={400} height={200} origin='/observed/03612600/00065' title={'Ohio at Olmsted: 03612600'} axis={"Gauge Height (ft)"}/>
            <StandardGraph width={400} height={200} origin='/observed/07022000/00065' title={'Mississippi at Thebes: 07022000'} axis={"Gauge Height (ft)"}/>
          </div>
        </div>
        <div className="Info">
          <p>Questions? Contact me at <a href='mailto:dberndt@aeci.org'>dberndt@aeci.org</a></p>
        </div>
      </div>
    </div>
  );
}

export default App;
