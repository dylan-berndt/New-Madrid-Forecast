import {useEffect, useState} from 'react';
import React from 'react';
import './style.css';
import error from '../Error.png';


export default function ErrorTooltip({height, text}) {
    if (text == "") return <></>;

    return <div className="ErrorBox" style={{height: height + "px"}}>
        <img src={error}></img>
        <p>{text}</p>
    </div>
}