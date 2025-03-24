import { LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Label, Legend} from 'recharts';
import {useEffect, useState, useRef} from 'react';
import React from 'react';
import './style.css';
import {formatDate, RangeSlider, CustomTooltip} from '../Tools';

function ToggleTooltip({toggles, setToggles}) {
    function toggle(name) {
        return () => {
            setToggles(prevToggles => ({
                ...prevToggles,
                [name]: !prevToggles[name]
            }));
        }
    }
    return <div class="ToggleTooltip">
        {Object.keys(toggles).map((key) => {
            return <>
            <button onClick={toggle(key)} style={{backgroundColor: toggles[key] ? "grey" : "white"}}></button>
            <p>{key}</p>
            </>
        })}
    </div>
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

    const xIndices = xValues.filter((element, index) => element >= left && element <= right);
    var chartData = xIndices.map((xIndex) => {
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

    const ticks = [...Array(5).keys()].map((x) => left + (x * (right - left)) / 4);

    return <div className="PredictionGraph">
        <h>{title}</h>
        <hr></hr>
        <LineChart className="PredictionLine" width={width} height={height} data={chartData}>
            <CartesianGrid stroke="#ccc" strokeDasharray="5 5" fillOpacity={0.0}/>

            {origins.map((key, index) => {
                if (!toggles[names[index]]) return;
                return <Line type="monotone" connectNulls dataKey={names[index]} stroke={colors[index]} dot={false} activeDot={true} strokeWidth={4} animationDuration={0}/>
            })}
            {/* <Line type="monotone" dataKey="y" stroke="#8884d8" dot={false} activeDot={true} strokeWidth={4}/> */}

            <XAxis dataKey="x" type="number" domain={[left, right]}  tickMargin={10} tickFormatter={formatDate} ticks={ticks}/>
            <YAxis>
                <Label value={axis} angle={-90} dx={-20}/>
            </YAxis>
            <Tooltip animationDuration={150} animationEasing='ease-out' content={CustomTooltip}/>
            <Legend verticalAlign="top" height={36}/>
        </LineChart>

        <RangeSlider setLeft={setLeft} setRight={setRight} maxLeft={maxLeft} maxRight={maxRight}/>
        <ToggleTooltip toggles={toggles} setToggles={setToggles}/>
    </div>
}