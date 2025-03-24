import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label} from 'recharts';
import {useEffect, useState} from 'react';
import React from 'react';
import './style.css';
import {formatDate, RangeSlider, CustomTooltip} from '../Tools';


export default function StandardGraph({width, height, origin, title, axis}) {
    const [timestamps, setTimestamps] = useState([]);
    const [data, setData] = useState([]);
    const [left, setLeft] = useState([-1]);
    const [right, setRight] = useState([-1]);

    const [maxLeft, setMaxLeft] = useState([-1]);
    const [maxRight, setMaxRight] = useState([-1]);

    useEffect(() => {
        fetch(origin)
        .then((response) => response.json())
        .then((data) => {
            setTimestamps(data.dates);
            setData(data.sequence);
        }) 
        .catch((err) => {
            console.error(err);
        })
    }, []);

    useEffect(() => {
        setLeft(Math.min(...timestamps));
        setRight(Math.max(...timestamps));
        setMaxLeft(Math.min(...timestamps));
        setMaxRight(Math.max(...timestamps));
    }, [data]);

    const dates = timestamps.filter((element, index) => element >= left && element <= right);
    const chartData = dates.map((element, index) => {
        const xIndex = timestamps.indexOf(element);
        var obj = {
            x: element
        }
        obj[axis] = data[xIndex];
        return obj;
    })

    chartData.sort((a, b) => a.x - b.x);

    const ticks = [...Array(5).keys()].map((x) => left + (x * (right - left)) / 4);

    return <div className="StandardGraph">
        <h>{title}</h>
        <hr></hr>
        <LineChart className="StandardLine" width={width} height={height} data={chartData}>
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" fillOpacity={0.0}/>
            <Line type="monotone" dataKey={axis} stroke="#8884d8" dot={false} activeDot={true} strokeWidth={4} animationDuration={0}/>
            <XAxis dataKey="x" type="number" domain={[left, right]}  tickMargin={10} tickFormatter={formatDate} ticks={ticks} >
                {/* <Label value={"Date"} offset={10} position="bottom"/> */}
            </XAxis>
            <YAxis>
                <Label value={axis} angle={-90} dx={-20}/>
            </YAxis>
            <Tooltip animationDuration={150} animationEasing='ease-out' content={CustomTooltip}/>
        </LineChart>

        <RangeSlider setLeft={setLeft} setRight={setRight} maxLeft={maxLeft} maxRight={maxRight}/>
    </div>
}