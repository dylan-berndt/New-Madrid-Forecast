import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label} from 'recharts';
import {useEffect, useState} from 'react';
import React from 'react';
import './style.css';

function RangeTooltip({setLeft, setRight}) {
    return <div className="RangeTooltip">

    </div>
}

export default function StandardGraph({width, height, origin, title, axis}) {
    const [timestamps, setTimestamps] = useState([1]);
    const [data, setData] = useState([1]);
    const [left, setLeft] = useState([-1]);
    const [right, setRight] = useState([-1]);

    // console.log("Data type:", typeof data);

    useEffect(() => {
        fetch(origin)
        .then((response) => response.json())
        .then((data) => {
            setData(data.sequence);
        }) 
        .catch((err) => {
            console.error(err);
        })
    }, []);

    useEffect(() => {
        setLeft(0);
        setRight(data.length);
    }, [data]);

    const lineData = data.filter((element, index) => index >= left && index <= right);
    const chartData = lineData.map((element, index) => {
        return {
            x: index,
            y: element
        }
    })

    return <div className="StandardGraph">
        <h>{title}</h>
        <hr></hr>
        <LineChart className="StandardLine" width={width} height={height} data={chartData}>
            <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} activeDot={true}/>
            <CartesianGrid stroke="#ccc" strokeDasharray="10 10"/>
            <XAxis dataKey="x" interval={14}>
                {/* <Label value={"Date"} offset={10} position="bottom"/> */}
            </XAxis>
            <YAxis>
                <Label value={axis} angle={-90} dx={-10}/>
            </YAxis>
            <Tooltip animationDuration={150} animationEasing='ease-out'/>
        </LineChart>

        <RangeTooltip setLeft={setLeft} setRight={setRight}/>
    </div>
}