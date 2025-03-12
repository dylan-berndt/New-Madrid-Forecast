import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label} from 'recharts';
import {useEffect, useState} from 'react';
import React from 'react';
import './style.css';

function RangeTooltip({setLeft, setRight}) {
    return <div className="RangeTooltip">

    </div>
}

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

    function formatDate(date) {
        const d = new Date(date * 1000);
        var minutes = d.getMinutes().toString();
        if (minutes.length == 1) {
            minutes = "0" + minutes;
        }
        var hours = d.getHours();
        const meridian = hours > 12 ? "AM" : "PM"
        hours = hours % 12;
        hours = hours == 0 ? 12 : hours;
        return (d.getMonth() + 1) + "/" + d.getDate() + " " + hours + ":" + minutes + " " + meridian;
    }

    // const lineData = data.filter((element, index) => index >= left && index <= right);
    // const dates = timestamps.filter((element, index) => index >= left && index <= right);
    const chartData = data.map((element, index) => {
        return {
            x: timestamps[index],
            y: element
        }
    })

    return <div className="StandardGraph">
        <h>{title}</h>
        <hr></hr>
        <LineChart className="StandardLine" width={width} height={height} data={chartData}>
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" fillOpacity={0.0}/>
            <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} activeDot={true} strokeWidth={4}/>
            <XAxis dataKey="x" type="number" domain={[left, right]}  tickMargin={10} tickFormatter={formatDate} >
                {/* <Label value={"Date"} offset={10} position="bottom"/> */}
            </XAxis>
            <YAxis>
                <Label value={axis} angle={-90} dx={-20}/>
            </YAxis>
            <Tooltip animationDuration={150} animationEasing='ease-out'/>
        </LineChart>

        <RangeTooltip setLeft={setLeft} setRight={setRight}/>
    </div>
}