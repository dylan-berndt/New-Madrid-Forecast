import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label, Legend} from 'recharts';
import {useEffect, useState} from 'react';
import React from 'react';
import './style.css';

function RangeTooltip({setLeft, setRight}) {
    return <div className="RangeTooltip">

    </div>
}

function ToggleTooltip({toggles, setToggles}) {
    return <></>
}

export default function PredictionGraph({width, height, origins, title, names, colors, axis}) {
    const [toggles, setToggles] = useState(names.reduce((obj, item, index) => {
        obj[item] = true;
        return obj;
    }, {}));

    const [series, setSeries] = useState(origins.reduce((obj, item, index) => {
        obj[item] = [];
        return obj;
    }, {}));

    const [timestamps, setTimestamps] = useState(origins.reduce((obj, item, index) => {
        obj[item] = [];
        return obj;
    }, {}));

    const [left, setLeft] = useState([-1]);
    const [right, setRight] = useState([-1]);

    const [maxLeft, setMaxLeft] = useState([-1]);
    const [maxRight, setMaxRight] = useState([-1]);

    const [xValues, setXValues] = useState([]);

    useEffect(() => {
        for (const origin of origins) {
            fetch(origin)
            .then((response) => response.json())
            .then((data) => {
                setSeries(prevSeries => ({
                    ...prevSeries,
                    [origin]: data.sequence
                }));
                setTimestamps(prevTimestamps => ({
                    ...prevTimestamps,
                    [origin]: data.dates
                }));
            })
            .catch((err) => {
                console.error(err);
            })
        }
    }, []);

    useEffect(() => {
        var values = [...new Set(Object.values(timestamps).flatMap(array => array))];
        setXValues(values);

        setLeft(Math.min(...values));
        setRight(Math.max(...values));
        setMaxLeft(Math.min(...values));
        setMaxRight(Math.max(...values));
    }, [series])


    if (series == null) {
        return <></>
    }

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

    // const xIndices = xValues.filter((element, index) => element >= left && element <= right);
    var chartData = xValues.map((xIndex) => {
        return origins.reduce((obj, origin, i) => {
            const originIndex = timestamps[origin].indexOf(xIndex);
            if (originIndex == -1) {
                obj[names[i]] = null;
            }
            else {
                obj[names[i]] = series[origin][originIndex];
            }
            return obj;
        }, {x: parseInt(xIndex)});
    });

    chartData.sort((a, b) => a.x - b.x);

    return <div className="PredictionGraph">
        <h>{title}</h>
        <hr></hr>
        <LineChart className="PredictionLine" width={width} height={height} data={chartData}>
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" fillOpacity={0.0}/>

            {origins.map((key, index) => {
                return <Line type="monotone" connectNulls dataKey={names[index]} stroke={colors[index]} dot={false} activeDot={true} strokeWidth={4}/>
            })}
            {/* <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} activeDot={true} strokeWidth={4}/> */}

            <XAxis dataKey="x" type="number" domain={[left, right]}  tickMargin={10} tickFormatter={formatDate}/>
            <YAxis>
                <Label value={axis} angle={-90} dx={-20}/>
            </YAxis>
            <Tooltip animationDuration={150} animationEasing='ease-out'/>
            <Legend verticalAlign="top" height={36}/>
        </LineChart>

        <RangeTooltip setLeft={setLeft} setRight={setRight}/>
        <ToggleTooltip toggles={toggles} setToggles={setToggles}/>
    </div>
}